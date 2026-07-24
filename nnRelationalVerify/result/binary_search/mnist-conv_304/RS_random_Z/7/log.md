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
execution time: IAR + LP analysis = 14.09 + 33.24 = 47.33 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.67 seconds, max iter: 100)

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
Binary search time: 209.88 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3342.80 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4287426, upper bound: 1.4297456
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4297458, upper bound: 1.4287424
time: 7.04 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.70 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.70
Output dim: 0, lower bound: -1.4287426, upper bound: 1.4297456
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.70
Output dim: 0, lower bound: -1.4297458, upper bound: 1.4287424

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3778286, 2.3716557
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9952402, 3.0040793
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3639655, 2.3552637
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7939110, 2.7859035
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7089901, 2.7149882
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0693145, 2.0756664
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3828316, 2.3934212
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5345812, 3.5058980
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7885613, 1.7947006
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4287363, upper bound: 1.4226339
time: 6.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4216276, upper bound: 1.4297414
time: 5.49 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3716555, 2.3750880
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -3.0001650, 2.9952397
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3552642, 2.3601241
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7859039, 2.7903666
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7123318, 2.7089896
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0728593, 2.0693145
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3887277, 2.3828316
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5058975, 3.5218782
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7919784, 1.7885611
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5732

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4294072, upper bound: 1.4287352
time: 7.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4297390, upper bound: 1.4284053
time: 6.27 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 27.86 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.86
Output dim: 0, lower bound: -1.4287363, upper bound: 1.4226339
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.86
Output dim: 0, lower bound: -1.4216276, upper bound: 1.4297414
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.86
Output dim: 0, lower bound: -1.4294072, upper bound: 1.4287352
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.86
Output dim: 0, lower bound: -1.4297390, upper bound: 1.4284053

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3759623, 2.3668416
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9905663, 3.0022650
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3641186, 2.3530860
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7926283, 2.7854056
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7043076, 2.7131643
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0653253, 2.0741131
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3772421, 2.3790150
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5324769, 3.5005293
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7847018, 1.7847772
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4287305, upper bound: 1.4219999
time: 6.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4210050, upper bound: 1.4220013
time: 8.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3730145, 2.3697891
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9934254, 2.9994059
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3617887, 2.3554168
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7934122, 2.7846217
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7071648, 2.7103066
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0677614, 2.0716772
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3684254, 2.3878317
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5292125, 3.5037937
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7786374, 1.7908409
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9776807

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5745

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4206308, upper bound: 1.4297263
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4216138, upper bound: 1.4287458
time: 7.43 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3743849, 2.3742568
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9982309, 3.0015917
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3595181, 2.3588314
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7861862, 2.7902803
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7133493, 2.7086797
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0716567, 2.0732656
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3865056, 2.3901305
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5200920, 3.5175543
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7911158, 1.7913918
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725

Time for backsubstitution: 14.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5814

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4286428, upper bound: 1.4287031
time: 7.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4293726, upper bound: 1.4279714
time: 5.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3708239, 2.3750880
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -3.0001650, 2.9933066
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3539705, 2.3601241
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7858171, 2.7903666
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7120218, 2.7089896
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0728593, 2.0681133
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3887277, 2.3806095
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5015745, 3.5218782
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7919784, 1.7876985
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4297191, upper bound: 1.4196398
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4209766, upper bound: 1.4283835
time: 5.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.49 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.49
Output dim: 0, lower bound: -1.4287305, upper bound: 1.4219999
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.49
Output dim: 0, lower bound: -1.4210050, upper bound: 1.4220013
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.49
Output dim: 0, lower bound: -1.4206308, upper bound: 1.4297263
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.49
Output dim: 0, lower bound: -1.4216138, upper bound: 1.4287458
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.49
Output dim: 0, lower bound: -1.4286428, upper bound: 1.4287031
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.49
Output dim: 0, lower bound: -1.4293726, upper bound: 1.4279714
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.49
Output dim: 0, lower bound: -1.4297191, upper bound: 1.4196398
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.49
Output dim: 0, lower bound: -1.4209766, upper bound: 1.4283835

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3659840, 2.3525138
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9753704, 2.9914975
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3659763, 2.3543947
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8013144, 2.7915306
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6864157, 2.7011724
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0617456, 2.0736227
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3390603, 2.3251042
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5222821, 3.4804287
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7680991, 1.7613313
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9566059, 2.9621582

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4287107, upper bound: 1.4132434
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4199669, upper bound: 1.4219798
time: 5.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3616347, 2.3568630
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9797993, 2.9870682
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3654270, 2.3549354
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7987528, 2.7925272
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6923170, 2.6952715
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0648322, 2.0705333
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3233314, 2.3408346
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5123763, 3.4903255
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7612555, 1.7681746
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9639616, 2.9548025

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4209852, upper bound: 1.4132469
time: 6.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4122488, upper bound: 1.4219833
time: 6.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3755732, 2.3689728
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9932747, 2.9998817
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3688135, 2.3531756
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8042011, 2.7811713
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7027435, 2.7241740
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0669560, 2.0742023
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3660855, 2.3951507
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5477085, 3.4978838
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7776027, 1.7940750
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9773374

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5732

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4202899, upper bound: 1.4297189
time: 22.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4206242, upper bound: 1.4293874
time: 6.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3721976, 2.3697891
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9934254, 2.9992552
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3595476, 2.3554168
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7899609, 2.7846217
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7071648, 2.7058859
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0677614, 2.0708716
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3684254, 2.3854914
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5233030, 3.5037937
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7786374, 1.7898066
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9776807

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5814

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4215939, upper bound: 1.4199819
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4128462, upper bound: 1.4287244
time: 6.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3736095, 2.3822737
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -3.0016394, 3.0012622
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3591614, 2.3625033
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7883778, 2.7900710
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7198505, 2.7080507
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0744820, 2.0729918
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3860669, 2.3947058
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5234699, 3.5172262
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7924664, 1.7912612
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5745

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4276472, upper bound: 1.4286870
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4286290, upper bound: 1.4277035
time: 14.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3743849, 2.3734818
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9979010, 3.0015917
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3595181, 2.3584745
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7859783, 2.7902803
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7127199, 2.7086797
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0713830, 2.0732656
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3865056, 2.3896923
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5197639, 3.5175543
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7909844, 1.7913918
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4293534, upper bound: 1.4192096
time: 5.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4206104, upper bound: 1.4279517
time: 5.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3526049, 2.3493927
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9953642, 2.9906840
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3464451, 2.3547940
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7972322, 2.7981706
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6954842, 2.6995745
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0754867, 2.0719504
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3820848, 2.3689685
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4786487, 3.5063047
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7840631, 1.7765195
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9762278, 2.9781725

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4297133, upper bound: 1.4123881
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4224704, upper bound: 1.4196342
time: 5.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3451281, 2.3568692
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9975424, 2.9885068
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3486404, 2.3525987
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7936215, 2.8017817
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7026062, 2.6924515
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0766964, 2.0707407
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3770866, 2.3739662
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4859996, 3.4989529
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7807992, 1.7797832
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9776335, 2.9781725

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4209708, upper bound: 1.4211338
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4137257, upper bound: 1.4283783
time: 5.18 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.32 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.32
Output dim: 0, lower bound: -1.4287107, upper bound: 1.4132434
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.32
Output dim: 0, lower bound: -1.4199669, upper bound: 1.4219798
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.32
Output dim: 0, lower bound: -1.4209852, upper bound: 1.4132469
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.32
Output dim: 0, lower bound: -1.4122488, upper bound: 1.4219833
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.32
Output dim: 0, lower bound: -1.4202899, upper bound: 1.4297189
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.32
Output dim: 0, lower bound: -1.4206242, upper bound: 1.4293874
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.32
Output dim: 0, lower bound: -1.4215939, upper bound: 1.4199819
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.32
Output dim: 0, lower bound: -1.4128462, upper bound: 1.4287244
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.32
Output dim: 0, lower bound: -1.4276472, upper bound: 1.4286870
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.32
Output dim: 0, lower bound: -1.4286290, upper bound: 1.4277035
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.32
Output dim: 0, lower bound: -1.4293534, upper bound: 1.4192096
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.32
Output dim: 0, lower bound: -1.4206104, upper bound: 1.4279517
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.32
Output dim: 0, lower bound: -1.4297133, upper bound: 1.4123881
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.32
Output dim: 0, lower bound: -1.4224704, upper bound: 1.4196342
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.32
Output dim: 0, lower bound: -1.4209708, upper bound: 1.4211338
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.32
Output dim: 0, lower bound: -1.4137257, upper bound: 1.4283783

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3477654, 2.3268185
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9705725, 2.9888763
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3584518, 2.3490655
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8127308, 2.7993369
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6698780, 2.6917577
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0643740, 2.0774612
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3324180, 2.3134642
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4993548, 3.4648523
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7601833, 1.7501523
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9517975, 2.9587541

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5732

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4283715, upper bound: 1.4132367
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4287039, upper bound: 1.4129050
time: 5.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3402886, 2.3342953
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9727488, 2.9866986
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3606472, 2.3468697
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8091202, 2.8029480
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6770010, 2.6846352
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0655842, 2.0762513
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3274198, 2.3184619
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5067058, 3.4575009
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7569199, 1.7534161
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9532022, 2.9573493

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4192024, upper bound: 1.4219461
time: 15.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4199324, upper bound: 1.4212158
time: 5.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3434157, 2.3311677
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9750013, 2.9844470
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3579025, 2.3496068
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8101702, 2.8003335
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6757793, 2.6858568
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0674610, 2.0743721
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3166890, 2.3291950
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4894490, 3.4747491
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7533398, 1.7569957
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9591522, 2.9513984

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4202210, upper bound: 1.4132124
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4209512, upper bound: 1.4124827
time: 5.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3359394, 2.3386445
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9771776, 2.9822698
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3600979, 2.3474109
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8065596, 2.8039446
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6829023, 2.6787338
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0686712, 2.0731621
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3116908, 2.3341928
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4967999, 3.4673977
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7500763, 1.7602594
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9605570, 2.9499936

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4114847, upper bound: 1.4219471
time: 7.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4122143, upper bound: 1.4212177
time: 5.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3783026, 2.3681409
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9913406, 3.0062284
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3730659, 2.3518829
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8044834, 2.7810855
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7037611, 2.7238646
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0657535, 2.0781507
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3638625, 2.4024415
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5618949, 3.4935579
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7767406, 1.7969074
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9768038

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5814

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4202701, upper bound: 1.4209578
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4115213, upper bound: 1.4296991
time: 7.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3747416, 2.3689728
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9932747, 2.9979477
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3675203, 2.3531756
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8041153, 2.7811713
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7024345, 2.7241740
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0669560, 2.0730000
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3660855, 2.3929276
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5433831, 3.4978838
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7776027, 1.7932122
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9773374

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4206043, upper bound: 1.4206253
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4118537, upper bound: 1.4293677
time: 5.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3539791, 2.3440938
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9886265, 2.9966331
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3520212, 2.3500872
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8013763, 2.7924252
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6906290, 2.6964712
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0703893, 2.0747092
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3617821, 2.3738503
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5003762, 3.4882183
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7707224, 1.7786274
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9742756

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5814

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5732

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4212545, upper bound: 1.4199752
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4215871, upper bound: 1.4196446
time: 6.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3465023, 2.3515706
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9908047, 2.9944553
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3542175, 2.3478918
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7977648, 2.7960358
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6977510, 2.6893482
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0715995, 2.0734992
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3567839, 2.3788476
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5077281, 3.4808674
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7674589, 1.7818911
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9728708

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4120811, upper bound: 1.4286902
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4128117, upper bound: 1.4279595
time: 6.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3761678, 2.3814559
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -3.0014858, 3.0017319
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3661757, 2.3602617
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7991652, 2.7866216
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7154284, 2.7219191
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0736766, 2.0755084
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3837252, 2.4020123
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5419388, 3.5113125
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7914317, 1.7944944
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4276409, upper bound: 1.4215716
time: 7.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4205326, upper bound: 1.4286830
time: 6.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3727922, 2.3822737
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -3.0016394, 3.0011091
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3569193, 2.3625033
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7849269, 2.7900710
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7198505, 2.7036290
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0744820, 2.0721855
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3860669, 2.3923645
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5175571, 3.5172262
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7924664, 1.7902265
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5814

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4286092, upper bound: 1.4189404
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4198676, upper bound: 1.4276843
time: 5.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3561654, 2.3477857
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9931021, 2.9989691
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3519926, 2.3531456
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7973938, 2.7980843
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6961803, 2.6992645
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0740118, 2.0771027
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3798628, 2.3780508
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4968386, 3.5019808
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7830694, 1.7802129
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9779930, 2.9781725

Time for backsubstitution: 14.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4293475, upper bound: 1.4119570
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4221047, upper bound: 1.4192037
time: 5.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3486886, 2.3552625
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9952803, 2.9967918
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3541880, 2.3509502
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.7937832, 2.8016953
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.7033033, 2.6921415
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0752215, 2.0758924
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3748646, 2.3830490
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.5041895, 3.4946289
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7798059, 1.7834764
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9781725, 2.9781725

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4206046, upper bound: 1.4207036
time: 7.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.4133595, upper bound: 1.4279458
time: 6.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.3426261, 2.3350651
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.9801674, 2.9799166
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.3469849, 2.3547928
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.8059187, 2.8042970
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.6775923, 2.6875844
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -2.0702806, 2.0698314
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.3439064, 2.3150597
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.4630861, 3.4808445
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.7674594, 1.7530732
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.9509773, 2.9626026

Time for backsubstitution: 14.22 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.3750882148742676
rel_dist={0: [-1.4297502739106367, 1.4297521247570089]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573862, upper bound: 1.0532065
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0532065, upper bound: 1.0573864
time: 5.96 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.46 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.46
Output dim: 0, lower bound: -1.0573862, upper bound: 1.0532065
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.46
Output dim: 0, lower bound: -1.0532065, upper bound: 1.0573864

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1073771, 2.1048918
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5627689, 2.5652995
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0370226, 2.0367134
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4773350, 2.4758716
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3162594, 2.3196321
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8013458, 1.8031096
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.1066756, 2.0976868
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0845509, 3.0788956
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5110509, 1.5071399
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7448597, 2.7490625

Time for backsubstitution: 13.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5732

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573193, upper bound: 1.0532054
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573847, upper bound: 1.0531397
time: 5.58 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1048918, 2.1073766
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5653000, 2.5627685
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0367136, 2.0370228
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4758720, 2.4773350
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3196321, 2.3162594
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8031092, 1.8013458
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0976872, 2.1066761
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0788956, 3.0845509
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5071399, 1.5110505
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7490625, 2.7448597

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 6123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524948, upper bound: 1.0524946
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524920, upper bound: 1.0573825
time: 5.12 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.78
Output dim: 0, lower bound: -1.0573193, upper bound: 1.0532054
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.78
Output dim: 0, lower bound: -1.0573847, upper bound: 1.0531397
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.78
Output dim: 0, lower bound: -1.0524948, upper bound: 1.0524946
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.78
Output dim: 0, lower bound: -1.0524920, upper bound: 1.0573825

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1085801, 2.1040602
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5608349, 2.5681005
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0388985, 2.0354192
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4774585, 2.4757848
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3167081, 2.3193221
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8001432, 1.8048511
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.1044536, 2.1009049
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0908089, 3.0745721
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5101879, 1.5083876
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7456331, 2.7485285

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573155, upper bound: 1.0524907
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524276, upper bound: 1.0524933
time: 6.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1065454, 2.1048918
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5627689, 2.5633659
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0357285, 2.0367134
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4772487, 2.4758716
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3159499, 2.3196321
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8013458, 1.8019071
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.1066756, 2.0954647
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0802279, 3.0788956
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5110509, 1.5062773
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7443256, 2.7490625

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573741, upper bound: 1.0482118
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524575, upper bound: 1.0531290
time: 5.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1017623, 2.1025634
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5606260, 2.5597291
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0371771, 2.0361550
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4745903, 2.4756064
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3149505, 2.3132110
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8007469, 1.8003750
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0883169, 2.0922680
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0807495, 3.0845394
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5006821, 1.5011275
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7451687, 2.7423234

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5732

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524279, upper bound: 1.0524935
time: 5.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524933, upper bound: 1.0524277
time: 5.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1000776, 2.1042476
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5622597, 2.5580955
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0358458, 2.0374911
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4750366, 2.4760528
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3165827, 2.3115783
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8021402, 1.7989829
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0832791, 2.0973053
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0788841, 3.0864100
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4972169, 1.5045924
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7465267, 2.7409658

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517434, upper bound: 1.0573626
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524722, upper bound: 1.0566342
time: 5.52 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.67 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.67
Output dim: 0, lower bound: -1.0573155, upper bound: 1.0524907
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.67
Output dim: 0, lower bound: -1.0524276, upper bound: 1.0524933
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.67
Output dim: 0, lower bound: -1.0573741, upper bound: 1.0482118
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.67
Output dim: 0, lower bound: -1.0524575, upper bound: 1.0531290
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.67
Output dim: 0, lower bound: -1.0524279, upper bound: 1.0524935
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.67
Output dim: 0, lower bound: -1.0524933, upper bound: 1.0524277
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.67
Output dim: 0, lower bound: -1.0517434, upper bound: 1.0573626
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.67
Output dim: 0, lower bound: -1.0524722, upper bound: 1.0566342

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1054511, 2.0992465
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5561638, 2.5650620
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0393677, 2.0345523
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4761767, 2.4749503
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3120265, 2.3162732
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7977800, 1.8038816
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0950823, 2.0864973
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0926671, 3.0745592
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5037301, 1.4984646
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7417393, 2.7459927

Time for backsubstitution: 14.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0568833, upper bound: 1.0524888
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573135, upper bound: 1.0520641
time: 5.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1037664, 2.1009312
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5577965, 2.5634284
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0380316, 2.0358841
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4757304, 2.4745026
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3136592, 2.3146410
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7991724, 1.8024883
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0900455, 2.0915351
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0907960, 3.0764246
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5002644, 1.5019295
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7430973, 2.7446346

Time for backsubstitution: 14.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5745

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516521, upper bound: 1.0524884
time: 7.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524223, upper bound: 1.0517180
time: 5.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0851212, 2.0791960
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5579691, 2.5598111
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0282040, 2.0304430
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4871144, 2.4836760
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.2994118, 2.3071647
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8039737, 1.8052273
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0978909, 2.0838227
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0572991, 3.0601683
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5017371, 1.4950984
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7395144, 2.7450562

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573703, upper bound: 1.0474979
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524824, upper bound: 1.0475006
time: 5.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0808487, 2.0834684
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5592136, 2.5585666
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0294590, 2.0291884
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4850526, 2.4857392
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3034821, 2.3030944
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8046651, 1.8045359
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0950346, 2.0866790
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0615001, 3.0559678
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4998717, 1.4969633
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7403173, 2.7442536

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520245, upper bound: 1.0531269
time: 8.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524556, upper bound: 1.0526956
time: 6.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1029658, 2.1017318
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5586948, 2.5625310
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0390539, 2.0348618
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4747138, 2.4755197
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3153987, 2.3129015
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7995443, 1.8021164
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0860949, 2.0954862
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0870061, 3.0802150
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4998190, 1.5023751
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7459421, 2.7417893

Time for backsubstitution: 14.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519982, upper bound: 1.0524930
time: 6.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524258, upper bound: 1.0520663
time: 5.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1009312, 2.1025634
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5606260, 2.5577970
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0358839, 2.0361550
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4745021, 2.4756064
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3146410, 2.3132110
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8007469, 1.7991724
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0883169, 2.0900455
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0764241, 3.0845394
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5006821, 1.5002646
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7446346, 2.7423234

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5745

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517181, upper bound: 1.0524223
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524880, upper bound: 1.0516521
time: 6.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0993028, 2.1084962
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5640659, 2.5577660
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0354900, 2.0394373
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4761982, 2.4758434
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3200274, 2.3109484
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8036375, 1.7987094
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0828404, 2.0997314
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0806742, 3.0860820
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4979334, 1.5044618
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7494259, 2.7404413

Time for backsubstitution: 14.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 6123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517328, upper bound: 1.0524367
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0468164, upper bound: 1.0573522
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1000776, 2.1034722
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5619297, 2.5580955
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0358458, 2.0371351
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4748278, 2.4760528
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3159533, 2.3115783
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8018665, 1.7989829
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0832791, 2.0968661
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0785561, 3.0864100
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4970865, 1.5045924
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7460022, 2.7409658

Time for backsubstitution: 14.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520454, upper bound: 1.0566337
time: 6.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524705, upper bound: 1.0562057
time: 6.28 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 27.45 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 0, lower bound: -1.0568833, upper bound: 1.0524888
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 0, lower bound: -1.0573135, upper bound: 1.0520641
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 0, lower bound: -1.0516521, upper bound: 1.0524884
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 0, lower bound: -1.0524223, upper bound: 1.0517180
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 0, lower bound: -1.0573703, upper bound: 1.0474979
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 0, lower bound: -1.0524824, upper bound: 1.0475006
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 0, lower bound: -1.0520245, upper bound: 1.0531269
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 0, lower bound: -1.0524556, upper bound: 1.0526956
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 0, lower bound: -1.0519982, upper bound: 1.0524930
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 0, lower bound: -1.0524258, upper bound: 1.0520663
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 0, lower bound: -1.0517181, upper bound: 1.0524223
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 0, lower bound: -1.0524880, upper bound: 1.0516521
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 0, lower bound: -1.0517328, upper bound: 1.0524367
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 0, lower bound: -1.0468164, upper bound: 1.0573522
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 0, lower bound: -1.0520454, upper bound: 1.0566337
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.45
Output dim: 0, lower bound: -1.0524705, upper bound: 1.0562057

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1055460, 2.0958138
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5512371, 2.5651846
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0394802, 2.0296919
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4762888, 2.4704881
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3086839, 2.3163586
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7942362, 1.8039677
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0891867, 2.0866494
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0930748, 3.0585785
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5003130, 1.4985576
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7417974, 2.7436624

Time for backsubstitution: 14.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0568726, upper bound: 1.0475616
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519498, upper bound: 1.0524782
time: 6.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1020184, 2.0992465
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5561638, 2.5601363
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0345078, 2.0345523
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4717150, 2.4749503
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3120265, 2.3129306
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7977800, 1.8003376
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0950823, 2.0806012
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0766859, 3.0745592
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5037301, 1.4950480
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7394094, 2.7459927

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5745

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0565375, upper bound: 1.0520586
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573083, upper bound: 1.0512858
time: 5.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1048779, 2.1001134
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5576448, 2.5636320
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0410795, 2.0336430
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4804173, 2.4710531
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3092370, 2.3206701
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7983670, 1.8035815
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0877032, 2.0947061
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0988121, 3.0705090
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4992297, 1.5033336
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7435617, 2.7442932

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0509034, upper bound: 1.0524689
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516323, upper bound: 1.0517394
time: 6.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1029491, 2.1009312
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5577965, 2.5632763
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0357904, 2.0358841
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4722824, 2.4745026
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3136592, 2.3102188
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7991724, 1.8016827
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0900455, 2.0891933
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0848799, 3.0764246
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5002644, 1.5008948
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7427559, 2.7446346

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516737, upper bound: 1.0516986
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524025, upper bound: 1.0509694
time: 5.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0819921, 2.0743818
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5532980, 2.5567727
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0286722, 2.0295756
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4858346, 2.4828429
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.2947307, 2.3041153
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8016105, 1.8042576
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0885201, 2.0694151
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0591564, 3.0601554
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4952788, 1.4851756
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7356215, 2.7425213

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5745

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0565949, upper bound: 1.0474923
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0573650, upper bound: 1.0467213
time: 5.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0803075, 2.0760660
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5549316, 2.5551391
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0273361, 2.0309074
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4853883, 2.4823952
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.2963634, 2.3024831
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8030028, 1.8028643
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0834832, 2.0744534
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0572863, 3.0620208
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4918137, 1.4886403
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7369795, 2.7411637

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520556, upper bound: 1.0474983
time: 6.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524805, upper bound: 1.0470668
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0809436, 2.0800357
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5542879, 2.5586934
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0295715, 2.0243270
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4851637, 2.4812760
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3001399, 2.3031797
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8011208, 1.8046210
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0891390, 2.0868344
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0619106, 3.0399880
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4964547, 1.4970551
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7403755, 2.7419214

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520207, upper bound: 1.0524126
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0471360, upper bound: 1.0524149
time: 7.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0774164, 2.0834684
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5592136, 2.5536423
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0245991, 2.0291884
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4805889, 2.4857392
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3034821, 2.2997522
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8046651, 1.8009915
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0950346, 2.0807834
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0455198, 3.0559678
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4998717, 1.4935467
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7379875, 2.7442536

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5745

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516795, upper bound: 1.0526904
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524504, upper bound: 1.0519155
time: 5.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1030607, 2.0982990
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5537682, 2.5626535
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0391674, 2.0300014
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4748259, 2.4710574
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3120561, 2.3129864
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7960000, 1.8022025
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0801988, 2.0956383
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0874138, 3.0642338
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4964025, 1.5024681
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7460003, 2.7394590

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512502, upper bound: 1.0524714
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0519784, upper bound: 1.0517414
time: 9.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0995331, 2.1017318
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5586948, 2.5576053
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0341940, 2.0348618
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4702520, 2.4755197
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3153987, 2.3095589
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7995443, 1.7985721
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0860949, 2.0895901
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0710249, 3.0802150
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4998190, 1.4989583
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7436123, 2.7417893

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0516772, upper bound: 1.0520465
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524060, upper bound: 1.0513183
time: 6.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1020422, 2.1017466
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5604763, 2.5580025
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0389347, 2.0339139
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4791889, 2.4721560
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3102188, 2.3192401
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7999406, 1.8002682
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0859752, 2.0932212
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0844498, 3.0786238
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4996474, 1.5016687
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7451010, 2.7419801

Time for backsubstitution: 14.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0512881, upper bound: 1.0524202
time: 9.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517160, upper bound: 1.0519926
time: 6.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.1001134, 2.1025634
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5606260, 2.5576448
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0336428, 2.0361550
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4710541, 2.4756064
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3146410, 2.3087888
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8007469, 1.7983668
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0883169, 2.0877037
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0705090, 3.0845394
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.5006821, 1.4992297
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7442932, 2.7423234

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0520612, upper bound: 1.0516498
time: 7.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0524866, upper bound: 1.0512182
time: 6.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0778794, 2.0828004
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5592690, 2.5542130
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0279655, 2.0331683
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4860668, 2.4836488
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3034892, 2.2984805
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8062658, 1.8020289
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0740557, 2.0880909
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0577450, 3.0673542
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4886191, 1.4932828
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7446156, 2.7364340

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0513065, upper bound: 1.0524344
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0517307, upper bound: 1.0520036
time: 6.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0736070, 2.0870728
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.5605125, 2.5529685
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -2.0292196, 2.0319128
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.4840040, 2.4857121
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.3075595, 2.2944102
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.8069572, 1.8013375
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0711999, 2.0909467
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -3.0619459, 3.0631533
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4867542, 1.4951477
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.7454176, 2.7356315

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 6123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5732

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0467492, upper bound: 1.0573503
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1.0468149, upper bound: 1.0572853
time: 6.99 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 28.68 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0568726, upper bound: 1.0475616
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0519498, upper bound: 1.0524782
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0565375, upper bound: 1.0520586
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0573083, upper bound: 1.0512858
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0509034, upper bound: 1.0524689
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0516323, upper bound: 1.0517394
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0516737, upper bound: 1.0516986
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0524025, upper bound: 1.0509694
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0565949, upper bound: 1.0474923
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0573650, upper bound: 1.0467213
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0520556, upper bound: 1.0474983
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0524805, upper bound: 1.0470668
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0520207, upper bound: 1.0524126
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0471360, upper bound: 1.0524149
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0516795, upper bound: 1.0526904
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0524504, upper bound: 1.0519155
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0512502, upper bound: 1.0524714
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0519784, upper bound: 1.0517414
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0516772, upper bound: 1.0520465
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0524060, upper bound: 1.0513183
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0512881, upper bound: 1.0524202
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0517160, upper bound: 1.0519926
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0520612, upper bound: 1.0516498
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0524866, upper bound: 1.0512182
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0513065, upper bound: 1.0524344
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0517307, upper bound: 1.0520036
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0467492, upper bound: 1.0573503
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 28.68
Output dim: 0, lower bound: -1.0468149, upper bound: 1.0572853
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 28.68
Output dim: 0, lower bound: -1.0520454, upper bound: 1.0566337
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 28.68
Output dim: 0, lower bound: -1.0524705, upper bound: 1.0562057
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.1192193031311035
rel_dist={0: [-1.0573893980579818, 1.057389512159725]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094649, upper bound: 0.9064660
time: 6.87 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9064686, upper bound: 0.9094641
time: 7.62 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.50 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.50
Output dim: 0, lower bound: -0.9094649, upper bound: 0.9064660
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.50
Output dim: 0, lower bound: -0.9064686, upper bound: 0.9094641

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0214658, 2.0196018
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4220352, 2.4239340
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9291420, 1.9289105
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3700962, 2.3689981
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1901994, 2.1927290
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7125769, 1.7138996
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0253844, 2.0186429
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9439621, 2.9397202
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4229429, 1.4200094
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6735954, 2.6767483

Time for backsubstitution: 14.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5732

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094315, upper bound: 0.9064654
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094643, upper bound: 0.9064329
time: 8.58 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0196018, 2.0214658
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4239349, 2.4220357
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9289103, 1.9291422
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3689976, 2.3700953
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1927285, 2.1901994
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7138996, 1.7125769
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0186429, 2.0253849
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9397202, 2.9439616
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4200094, 1.4229424
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6767483, 2.6735959

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9057008, upper bound: 0.9094481
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9064480, upper bound: 0.9086984
time: 5.67 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 25.76 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.76
Output dim: 0, lower bound: -0.9094315, upper bound: 0.9064654
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.76
Output dim: 0, lower bound: -0.9094643, upper bound: 0.9064329
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 25.76
Output dim: 0, lower bound: -0.9057008, upper bound: 0.9094481
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 25.76
Output dim: 0, lower bound: -0.9064480, upper bound: 0.9086984

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0221605, 2.0187707
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4201012, 2.4255509
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9302263, 1.9276164
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3701663, 2.3689113
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1904583, 2.1924191
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7113743, 1.7149050
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0231624, 2.0205011
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9475746, 2.9353971
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4220798, 1.4207296
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6740427, 2.6762137

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094235, upper bound: 0.9027963
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9057623, upper bound: 0.9064576
time: 5.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0206347, 2.0196018
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4220352, 2.4220004
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9278479, 1.9289105
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3700080, 2.3689981
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1898899, 2.1927290
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7125769, 1.7126970
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0253844, 2.0164204
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9396381, 2.9397202
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4229429, 1.4191468
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6730614, 2.6767483

Time for backsubstitution: 14.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094563, upper bound: 0.9027635
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9057951, upper bound: 0.9064248
time: 6.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0188265, 2.0244584
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4252062, 2.4217062
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9285545, 1.9305134
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3698168, 2.3698864
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1951547, 2.1895695
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7149544, 1.7123032
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0182042, 2.0270948
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9409814, 2.9436350
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4205139, 1.4228117
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6787910, 2.6730714

Time for backsubstitution: 14.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054896, upper bound: 0.9094451
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9057003, upper bound: 0.9092344
time: 5.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0196018, 2.0206904
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4236050, 2.4220357
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9289103, 1.9287868
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3687887, 2.3700953
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1920991, 2.1901994
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7136259, 1.7125769
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0186429, 2.0249462
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9393935, 2.9439616
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4198787, 1.4229424
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6762238, 2.6735959

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9064402, upper bound: 0.9050316
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9027797, upper bound: 0.9086905
time: 7.10 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 28.51 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.51
Output dim: 0, lower bound: -0.9094235, upper bound: 0.9027963
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 28.51
Output dim: 0, lower bound: -0.9057623, upper bound: 0.9064576
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.51
Output dim: 0, lower bound: -0.9094563, upper bound: 0.9027635
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 28.51
Output dim: 0, lower bound: -0.9057951, upper bound: 0.9064248
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.51
Output dim: 0, lower bound: -0.9054896, upper bound: 0.9094451
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.51
Output dim: 0, lower bound: -0.9057003, upper bound: 0.9092344
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 28.51
Output dim: 0, lower bound: -0.9064402, upper bound: 0.9050316
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.51
Output dim: 0, lower bound: -0.9027797, upper bound: 0.9086905

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9996681, 1.9930739
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4153032, 2.4216847
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9227009, 1.9210327
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3795190, 2.3767147
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1739202, 2.1789341
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7140026, 1.7180524
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0136623, 2.0088592
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9246459, 2.9156189
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4122996, 1.4095509
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6692314, 2.6720047

Time for backsubstitution: 14.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092123, upper bound: 0.9027959
time: 6.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094230, upper bound: 0.9025850
time: 6.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9981418, 1.9939065
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4172363, 2.4181342
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9203234, 1.9223263
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3793607, 2.3768020
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1733518, 2.1792440
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7152047, 1.7158444
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0158854, 2.0047789
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9167104, 2.9199433
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4131627, 1.4079678
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6682510, 2.6725407

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5745

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9087123, upper bound: 0.9027586
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094515, upper bound: 0.9020194
time: 6.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0180397, 2.0210261
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4202805, 2.4205699
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9274240, 1.9256525
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3687849, 2.3654232
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1918130, 2.1887984
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7114096, 1.7114811
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0123081, 2.0257368
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9372940, 2.9276543
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4170969, 1.4220264
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6782522, 2.6707411

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9048650, upper bound: 0.9058218
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9048630, upper bound: 0.9094424
time: 8.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0153942, 2.0236716
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4240704, 2.4167814
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9236932, 1.9293818
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3653536, 2.3688545
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1943836, 2.1862273
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7141323, 1.7087588
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0168462, 2.0211983
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9250011, 2.9399471
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4197280, 1.4193950
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6764612, 2.6725321

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9050758, upper bound: 0.9056112
time: 8.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9050737, upper bound: 0.9092318
time: 6.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9939065, 1.9981999
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4197369, 2.4172363
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9223266, 1.9212618
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3765926, 2.3794475
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1786137, 2.1736617
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7167735, 1.7152047
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0070019, 2.0154467
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9196157, 2.9210339
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4087005, 1.4131625
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6720142, 2.6687870

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5745

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9020356, upper bound: 0.9086854
time: 6.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9027750, upper bound: 0.9079466
time: 6.00 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 27.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.46
Output dim: 0, lower bound: -0.9092123, upper bound: 0.9027959
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.46
Output dim: 0, lower bound: -0.9094230, upper bound: 0.9025850
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.46
Output dim: 0, lower bound: -0.9087123, upper bound: 0.9027586
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.46
Output dim: 0, lower bound: -0.9094515, upper bound: 0.9020194
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 27.46
Output dim: 0, lower bound: -0.9048650, upper bound: 0.9058218
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.46
Output dim: 0, lower bound: -0.9048630, upper bound: 0.9094424
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 27.46
Output dim: 0, lower bound: -0.9050758, upper bound: 0.9056112
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.46
Output dim: 0, lower bound: -0.9050737, upper bound: 0.9092318
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.46
Output dim: 0, lower bound: -0.9020356, upper bound: 0.9086854
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 27.46
Output dim: 0, lower bound: -0.9027750, upper bound: 0.9079466

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9988818, 1.9896412
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4103785, 2.4205470
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9215708, 1.9161723
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3784866, 2.3722520
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1705780, 2.1781626
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7104583, 1.7172308
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0077672, 2.0075002
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9209576, 2.8996391
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4088831, 1.4087663
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6686926, 2.6696749

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5745

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9084683, upper bound: 0.9027907
time: 6.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092075, upper bound: 0.9020517
time: 6.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9962358, 1.9922867
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4141665, 2.4167604
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9178410, 1.9199021
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3750553, 2.3756833
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1731482, 2.1755919
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7131805, 1.7145081
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0123057, 2.0029640
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9086666, 2.9119325
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4115143, 1.4061341
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6669016, 2.6714659

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5745

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086790, upper bound: 0.9025799
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094183, upper bound: 0.9018435
time: 7.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9987721, 1.9930892
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4170847, 2.4182501
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9220510, 1.9200847
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3820114, 2.3733511
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1689310, 2.1826615
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7143993, 1.7164645
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0135436, 2.0065756
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9212523, 2.9140282
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4121270, 1.4087627
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6685133, 2.6721964

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9087097, upper bound: 0.9021320
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9050892, upper bound: 0.9021337
time: 6.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9973259, 1.9939065
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4172363, 2.4179816
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9180827, 1.9223263
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3759098, 2.3768020
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1733518, 2.1748233
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7152047, 1.7150388
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0158854, 2.0024376
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9107962, 2.9199433
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4131627, 1.4069335
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6679077, 2.6725407

Time for backsubstitution: 14.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094486, upper bound: 0.9013928
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9058281, upper bound: 0.9013972
time: 6.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0132256, 2.0174751
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4168339, 2.4158974
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9265556, 1.9257874
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3678389, 2.3641405
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1883554, 2.1841164
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7100921, 1.7091184
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -1.9979000, 2.0151067
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9372811, 2.9290447
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4071746, 1.4147019
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6753769, 2.6668472

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9048549, upper bound: 0.9057741
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9011937, upper bound: 0.9094351
time: 5.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -2.0105801, 2.0201206
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4206228, 2.4121094
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9228268, 1.9295168
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3644075, 2.3675718
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1909261, 2.1815457
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7128143, 1.7063961
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0024385, 2.0105681
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9249883, 2.9413376
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4098058, 1.4120705
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6735859, 2.6686382

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4575
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4575

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9050657, upper bound: 0.9055633
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9014045, upper bound: 0.9092239
time: 5.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9945359, 1.9973826
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4195862, 2.4173527
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9240532, 1.9190197
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3792443, 2.3759956
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1741929, 2.1770797
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7159677, 1.7158251
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0046601, 2.0172424
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9241562, 2.9151192
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4076648, 1.4139566
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6722765, 2.6684427

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 6123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9014110, upper bound: 0.9050624
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9014090, upper bound: 0.9086852
time: 6.38 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.73
Output dim: 0, lower bound: -0.9084683, upper bound: 0.9027907
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.73
Output dim: 0, lower bound: -0.9092075, upper bound: 0.9020517
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.73
Output dim: 0, lower bound: -0.9086790, upper bound: 0.9025799
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.73
Output dim: 0, lower bound: -0.9094183, upper bound: 0.9018435
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.73
Output dim: 0, lower bound: -0.9087097, upper bound: 0.9021320
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 26.73
Output dim: 0, lower bound: -0.9050892, upper bound: 0.9021337
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.73
Output dim: 0, lower bound: -0.9094486, upper bound: 0.9013928
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 26.73
Output dim: 0, lower bound: -0.9058281, upper bound: 0.9013972
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.73
Output dim: 0, lower bound: -0.9048549, upper bound: 0.9057741
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.73
Output dim: 0, lower bound: -0.9011937, upper bound: 0.9094351
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.73
Output dim: 0, lower bound: -0.9050657, upper bound: 0.9055633
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.73
Output dim: 0, lower bound: -0.9014045, upper bound: 0.9092239
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.73
Output dim: 0, lower bound: -0.9014110, upper bound: 0.9050624
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.73
Output dim: 0, lower bound: -0.9014090, upper bound: 0.9086852

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9980650, 1.9896412
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4103785, 2.4203944
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9193296, 1.9161723
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3750358, 2.3722520
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1705780, 2.1737413
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7104583, 1.7164252
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0077672, 2.0051589
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9150457, 2.8996391
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4088831, 1.4077313
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6683512, 2.6696749

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092046, upper bound: 0.9014251
time: 7.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9055841, upper bound: 0.9014271
time: 6.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9968657, 1.9914703
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4140148, 2.4168749
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9195662, 1.9176605
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3777080, 2.3722334
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1687269, 2.1790090
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7123752, 1.7151265
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0099645, 2.0047579
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9132032, 2.9060197
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4104791, 1.4069283
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6671648, 2.6711245

Time for backsubstitution: 14.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086764, upper bound: 0.9019532
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9050559, upper bound: 0.9019551
time: 6.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9954190, 1.9922867
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4141665, 2.4166079
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9155998, 1.9199021
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3716044, 2.3756833
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1731482, 2.1711702
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7131805, 1.7137024
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0123057, 2.0006227
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9027538, 2.9119325
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4115143, 1.4050992
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6665602, 2.6714659

Time for backsubstitution: 14.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094154, upper bound: 0.9012144
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9057949, upper bound: 0.9012189
time: 6.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9952221, 1.9882746
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4124136, 2.4148040
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9221854, 1.9192173
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3807316, 2.3724060
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1642489, 2.1792035
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7120361, 1.7151470
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0029140, 1.9921679
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9226413, 2.9140162
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4048033, 1.3988397
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6646204, 2.6693220

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9084985, upper bound: 0.9021337
time: 8.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9087093, upper bound: 0.9019206
time: 6.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9937754, 1.9890919
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4125652, 2.4145360
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9182172, 1.9214590
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3746281, 2.3758574
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1686707, 2.1713653
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7128415, 1.7137210
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -2.0052557, 1.9880300
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9121852, 2.9199305
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4058380, 1.3970106
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6640158, 2.6696663

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6123
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092373, upper bound: 0.9013923
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094481, upper bound: 0.9011840
time: 7.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9875307, 1.9949846
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4129677, 2.4110985
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9199724, 1.9182625
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3756437, 2.3734937
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1748695, 2.1675782
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7132397, 1.7117476
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -1.9862595, 2.0056081
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9175034, 2.9061165
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.3959949, 1.4049213
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6711674, 2.6620364

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5745
type: RSZ, layer: 1, pos: 5732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5745

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9004497, upper bound: 0.9094297
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9011886, upper bound: 0.9086931
time: 6.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9848852, 1.9976301
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4167557, 2.4073105
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9162426, 1.9219918
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3722124, 2.3769245
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1774406, 2.1650076
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7159619, 1.7090254
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -1.9907975, 2.0010695
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9052105, 2.9184093
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.3986261, 1.4022899
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6693764, 2.6638274

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5732

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9013711, upper bound: 0.9092233
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9014039, upper bound: 0.9091908
time: 10.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9897213, 1.9938316
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4161396, 2.4126816
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9231863, 1.9191546
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3782992, 2.3747149
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1707349, 2.1723971
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7146497, 1.7134619
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -1.9902525, 2.0066128
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9241443, 2.9165101
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.3977420, 1.4066322
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6694021, 2.6645498

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5732
type: RSZ, layer: 1, pos: 6123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5732

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9013757, upper bound: 0.9086821
time: 6.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9014085, upper bound: 0.9086491
time: 7.83 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 29.19 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.19
Output dim: 0, lower bound: -0.9092046, upper bound: 0.9014251
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 29.19
Output dim: 0, lower bound: -0.9055841, upper bound: 0.9014271
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.19
Output dim: 0, lower bound: -0.9086764, upper bound: 0.9019532
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 29.19
Output dim: 0, lower bound: -0.9050559, upper bound: 0.9019551
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.19
Output dim: 0, lower bound: -0.9094154, upper bound: 0.9012144
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 29.19
Output dim: 0, lower bound: -0.9057949, upper bound: 0.9012189
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 29.19
Output dim: 0, lower bound: -0.9084985, upper bound: 0.9021337
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.19
Output dim: 0, lower bound: -0.9087093, upper bound: 0.9019206
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.19
Output dim: 0, lower bound: -0.9092373, upper bound: 0.9013923
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.19
Output dim: 0, lower bound: -0.9094481, upper bound: 0.9011840
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.19
Output dim: 0, lower bound: -0.9004497, upper bound: 0.9094297
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.19
Output dim: 0, lower bound: -0.9011886, upper bound: 0.9086931
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.19
Output dim: 0, lower bound: -0.9013711, upper bound: 0.9092233
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.19
Output dim: 0, lower bound: -0.9014039, upper bound: 0.9091908
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.19
Output dim: 0, lower bound: -0.9013757, upper bound: 0.9086821
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.19
Output dim: 0, lower bound: -0.9014085, upper bound: 0.9086491

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: 7.7540379, 10.2373838, 7.7540379, 10.2373838, -1.9945145, 1.9848275
1: -19.2597141, -15.2714071, -19.2597141, -15.2714071, -2.4057074, 2.4169474
2: -6.5238490, -3.5489070, -6.5238490, -3.5489070, -1.9194641, 1.9153054
3: -10.8192272, -7.7928076, -10.8192272, -7.7928076, -2.3737559, 2.3713069
4: -13.5905094, -10.5921249, -13.5905094, -10.5921249, -2.1658964, 2.1702833
5: -4.6404066, -2.1593928, -4.6404066, -2.1593928, -1.7080960, 1.7151079
6: -4.5149174, -1.9158837, -4.5149174, -1.9158837, -1.9971375, 1.9907517
7: -12.8235607, -8.7824364, -12.8235607, -8.7824364, -2.9164362, 2.8996272
8: -5.4501801, -3.1462440, -5.4501801, -3.1462440, -1.4015589, 1.3978086
9: -1.9316568, 1.0465157, -1.9316568, 1.0465157, -2.6644583, 2.6668005

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9084387, upper bound: 0.9014074
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9091861, upper bound: 0.9006590
time: 5.38 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 25.42 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 25.42
Output dim: 0, lower bound: -0.9084387, upper bound: 0.9014074
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 25.42
Output dim: 0, lower bound: -0.9091861, upper bound: 0.9006590
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.42
Output dim: 0, lower bound: -0.9086764, upper bound: 0.9019532
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.42
Output dim: 0, lower bound: -0.9094154, upper bound: 0.9012144
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.42
Output dim: 0, lower bound: -0.9087093, upper bound: 0.9019206
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.42
Output dim: 0, lower bound: -0.9092373, upper bound: 0.9013923
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.42
Output dim: 0, lower bound: -0.9094481, upper bound: 0.9011840
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.42
Output dim: 0, lower bound: -0.9004497, upper bound: 0.9094297
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.42
Output dim: 0, lower bound: -0.9011886, upper bound: 0.9086931
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.42
Output dim: 0, lower bound: -0.9013711, upper bound: 0.9092233
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.42
Output dim: 0, lower bound: -0.9014039, upper bound: 0.9091908
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 25.42
Output dim: 0, lower bound: -0.9013757, upper bound: 0.9086821
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 25.42
Output dim: 0, lower bound: -0.9014085, upper bound: 0.9086491
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.0339293479919434
rel_dist={0: [-0.9094671207244254, 0.9094665870331546]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2431.32 seconds
