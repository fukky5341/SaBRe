## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.281490264


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1638608, 1.1638610)
1: (3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5360075, 0.5360075)
2: (-4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5615702, 0.5615700)
3: (-12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.8178110, 0.8178110)
4: (-2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7643485, 0.7643486)
5: (-9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5886670, 0.5886672)
6: (-7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8692248, 0.8692250)
7: (-2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3831897, 0.3831897)
8: (-3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6626787, 0.6626787)
9: (-12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7445683, 0.7445687)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.12 + 35.15 = 56.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.2843336, upper bound: 0.2843319

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6193
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 4599
type: DSZ, layer: 1, pos: 6155
type: DSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6193

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2843297, upper bound: 0.2805591
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2805577, upper bound: 0.2843311
time: 4.30 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.80 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.80
Output dim: 1, lower bound: -0.2843297, upper bound: 0.2805591
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.80
Output dim: 1, lower bound: -0.2805577, upper bound: 0.2843311

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1530905, 1.1509461
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5278484, 0.5262161
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5395142, 0.5351100
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.8291030, 0.8264871
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7685044, 0.7697606
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5951849, 0.5936751
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8590246, 0.8569951
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3675946, 0.3643711
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6745183, 0.6803833
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7374091, 0.7359817

Time for backsubstitution: 20.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4599
type: DSZ, layer: 1, pos: 6155
type: DSZ, layer: 1, pos: 5815
type: DSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4599

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2843227, upper bound: 0.2791272
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2791273, upper bound: 0.2805521
time: 3.55 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1509461, 1.1530902
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5262161, 0.5278484
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5351101, 0.5395142
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.8264871, 0.8291030
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7697606, 0.7685044
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5936753, 0.5951848
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8569952, 0.8590248
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3643711, 0.3675946
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6803832, 0.6745185
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7359815, 0.7374091

Time for backsubstitution: 20.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4599
type: DSZ, layer: 1, pos: 5815
type: DSZ, layer: 1, pos: 6155
type: DSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4599

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2805506, upper bound: 0.2828994
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2791273, upper bound: 0.2843237
time: 3.33 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.15 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.15
Output dim: 1, lower bound: -0.2843227, upper bound: 0.2791272
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 27.15
Output dim: 1, lower bound: -0.2791273, upper bound: 0.2805521
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.15
Output dim: 1, lower bound: -0.2805506, upper bound: 0.2828994
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.15
Output dim: 1, lower bound: -0.2791273, upper bound: 0.2843237

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1535935, 1.1515872
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5232959, 0.5207586
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5153196, 0.5149344
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.8298707, 0.8256490
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7607098, 0.7632586
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5893712, 0.5866992
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8420484, 0.8366237
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3682600, 0.3665301
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6759813, 0.6806368
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7390397, 0.7372670

Time for backsubstitution: 20.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 6155
type: DSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 901

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2843205, upper bound: 0.2755110
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2806946, upper bound: 0.2791240
time: 3.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1514497, 1.1537313
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5216637, 0.5223908
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5109150, 0.5193386
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.8272548, 0.8282650
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7619658, 0.7620026
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5878615, 0.5882092
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8400190, 0.8386533
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3650365, 0.3697536
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6818464, 0.6747718
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7376120, 0.7386944

Time for backsubstitution: 20.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5815
type: DSZ, layer: 1, pos: 6155
type: DSZ, layer: 1, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5815

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2805456, upper bound: 0.2780530
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2757117, upper bound: 0.2828946
time: 3.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1515870, 1.1535933
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5207587, 0.5232959
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5149343, 0.5153195
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.8256490, 0.8298709
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7632585, 0.7607099
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5866995, 0.5893712
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8366237, 0.8420484
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3665301, 0.3682601
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6806366, 0.6759813
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7372670, 0.7390397

Time for backsubstitution: 20.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 5815
type: DSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 901

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2755111, upper bound: 0.2806960
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2755111, upper bound: 0.2843218
time: 3.42 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 27.75 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.75
Output dim: 1, lower bound: -0.2843205, upper bound: 0.2755110
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 27.75
Output dim: 1, lower bound: -0.2806946, upper bound: 0.2791240
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 27.75
Output dim: 1, lower bound: -0.2805456, upper bound: 0.2780530
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.75
Output dim: 1, lower bound: -0.2757117, upper bound: 0.2828946
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 27.75
Output dim: 1, lower bound: -0.2755111, upper bound: 0.2806960
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.75
Output dim: 1, lower bound: -0.2755111, upper bound: 0.2843218

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1521473, 1.1475818
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5232957, 0.5208448
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5145411, 0.5127718
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.8249812, 0.8238964
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7607081, 0.7642070
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5877798, 0.5861273
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8416528, 0.8355491
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3682007, 0.3658626
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6757290, 0.6799371
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7389531, 0.7372283

Time for backsubstitution: 20.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6155
type: DSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6155

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2809854, upper bound: 0.2755027
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2843122, upper bound: 0.2721695
time: 4.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1393168, 1.1436133
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5169249, 0.5184386
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.4964730, 0.5073009
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.7689154, 0.7582848
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7604377, 0.7613969
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5772917, 0.5755267
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8340132, 0.8336391
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3645980, 0.3694878
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6825812, 0.6753318
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7388277, 0.7402899

Time for backsubstitution: 21.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 901

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2750715, upper bound: 0.2774148
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2736426, upper bound: 0.2828921
time: 3.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1475816, 1.1521473
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5208448, 0.5232959
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5127718, 0.5145411
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.8238964, 0.8249812
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7642069, 0.7607082
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5861273, 0.5877800
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8355492, 0.8416529
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3658626, 0.3682008
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6799371, 0.6757290
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7372279, 0.7389530

Time for backsubstitution: 21.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5815
type: DSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5815

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2736448, upper bound: 0.2788450
time: 3.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2736412, upper bound: 0.2843169
time: 3.43 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.32 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.32
Output dim: 1, lower bound: -0.2809854, upper bound: 0.2755027
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.32
Output dim: 1, lower bound: -0.2843122, upper bound: 0.2721695
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.32
Output dim: 1, lower bound: -0.2750715, upper bound: 0.2774148
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.32
Output dim: 1, lower bound: -0.2736426, upper bound: 0.2828921
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.32
Output dim: 1, lower bound: -0.2736448, upper bound: 0.2788450
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.32
Output dim: 1, lower bound: -0.2736412, upper bound: 0.2843169

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1512637, 1.1462405
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5164142, 0.5104071
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5135369, 0.5121098
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.8134599, 0.8162990
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7490990, 0.7565502
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5626712, 0.5479921
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8241113, 0.8089204
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3586560, 0.3595729
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6530998, 0.6650233
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7245841, 0.7153865

Time for backsubstitution: 20.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5815

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5815

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2843074, upper bound: 0.2703075
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2788354, upper bound: 0.2703111
time: 3.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1353118, 1.1421738
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5170109, 0.5184383
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.4943109, 0.5065237
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.7671649, 0.7533958
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7603271, 0.7603362
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5767200, 0.5739355
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8329387, 0.8332515
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3644834, 0.3699813
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6818818, 0.6750798
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7387886, 0.7402031

Time for backsubstitution: 20.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6155

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2703075, upper bound: 0.2828840
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2750577, upper bound: 0.2795572
time: 3.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1354496, 1.1420357
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5161058, 0.5193434
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.4983301, 0.5025046
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.7655592, 0.7550018
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7616198, 0.7590437
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5755579, 0.5750976
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8295436, 0.8366468
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3659769, 0.3684877
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6806723, 0.6762893
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7384436, 0.7405484

Time for backsubstitution: 21.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6155

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2703061, upper bound: 0.2843087
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2736329, upper bound: 0.2755084
time: 5.63 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 30.24 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.24
Output dim: 1, lower bound: -0.2843074, upper bound: 0.2703075
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.24
Output dim: 1, lower bound: -0.2788354, upper bound: 0.2703111
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.24
Output dim: 1, lower bound: -0.2703075, upper bound: 0.2828840
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.24
Output dim: 1, lower bound: -0.2750577, upper bound: 0.2795572
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.24
Output dim: 1, lower bound: -0.2703061, upper bound: 0.2843087
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.24
Output dim: 1, lower bound: -0.2736329, upper bound: 0.2755084

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1411521, 1.1341085
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5124618, 0.5056683
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5015000, 0.4976677
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.7434802, 0.7579613
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7474339, 0.7539623
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5499885, 0.5374223
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8191047, 0.8029143
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3589429, 0.3596871
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6536599, 0.6657584
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7261796, 0.7166021

Time for backsubstitution: 21.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1188
type: DSZ, layer: 3, pos: 1971
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 662
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 325

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2321

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2734495, upper bound: 0.2598379
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2734365, upper bound: 0.2588607
time: 3.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1339710, 1.1412902
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5065734, 0.5115569
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.4936486, 0.5055192
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.7595673, 0.7418747
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7526696, 0.7487266
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5385845, 0.5488266
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8063095, 0.8157092
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3581936, 0.3604364
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6669680, 0.6524503
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7169471, 0.7258344

Time for backsubstitution: 21.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 1971
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 325
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 1188
type: DSZ, layer: 3, pos: 662
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 2578

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 403

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2697289, upper bound: 0.2805686
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2694036, upper bound: 0.2808932
time: 3.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1341083, 1.1411524
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5056682, 0.5124619
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.4976677, 0.5015000
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.7579613, 0.7434802
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7539623, 0.7474339
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5374224, 0.5499886
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8029144, 0.8191046
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3596871, 0.3589429
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6657583, 0.6536598
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7166021, 0.7261796

Time for backsubstitution: 20.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 325
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 698
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 662
type: DSZ, layer: 3, pos: 1971
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 1188
type: DSZ, layer: 3, pos: 2327

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1990

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2665672, upper bound: 0.2804279
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2663663, upper bound: 0.2805873
time: 3.87 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 28.80 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 28.80
Output dim: 1, lower bound: -0.2734495, upper bound: 0.2598379
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 28.80
Output dim: 1, lower bound: -0.2734365, upper bound: 0.2588607
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 28.80
Output dim: 1, lower bound: -0.2697289, upper bound: 0.2805686
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 28.80
Output dim: 1, lower bound: -0.2694036, upper bound: 0.2808932
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 28.80
Output dim: 1, lower bound: -0.2665672, upper bound: 0.2804279
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 28.80
Output dim: 1, lower bound: -0.2663663, upper bound: 0.2805873

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 56.27 + 400.22 = 456.49 seconds
