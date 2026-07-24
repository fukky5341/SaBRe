## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.16436890799999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4758203, 0.4758205)
1: (-10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4666057, 0.4666057)
2: (-8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4128773, 0.4128771)
3: (-10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3284395, 0.3284395)
4: (-9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3763745, 0.3763745)
5: (7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3506265, 0.3506265)
6: (-4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3361690, 0.3361690)
7: (-13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4507301, 0.4507303)
8: (0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2773076, 0.2773077)
9: (-6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4804311, 0.4804308)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.71 + 33.78 = 58.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.1660289, upper bound: 0.1660293

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6165
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 836

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6165

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660230, upper bound: 0.1660166
time: 4.38 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660162, upper bound: 0.1660234
time: 2.97 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.37 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.37
Output dim: 5, lower bound: -0.1660230, upper bound: 0.1660166
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.37
Output dim: 5, lower bound: -0.1660162, upper bound: 0.1660234

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4778798, 0.4757769
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4665823, 0.4677279
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4128542, 0.4140124
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3288717, 0.3284317
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3763847, 0.3763745
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3508117, 0.3506227
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3361654, 0.3363371
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4507415, 0.4507294
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2780521, 0.2772918
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4818778, 0.4804003

Time for backsubstitution: 22.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 836

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660181, upper bound: 0.1638359
time: 4.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1638422, upper bound: 0.1660114
time: 4.18 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4757767, 0.4758205
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4666057, 0.4665823
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4128773, 0.4128540
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3284317, 0.3284395
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3763742, 0.3763745
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3506227, 0.3506265
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3361690, 0.3361654
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4507296, 0.4507303
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2772918, 0.2773077
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4804001, 0.4804308

Time for backsubstitution: 23.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 836
type: DSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 836

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660114, upper bound: 0.1638427
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1638355, upper bound: 0.1660186
time: 4.10 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.57 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.57
Output dim: 5, lower bound: -0.1660181, upper bound: 0.1638359
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.57
Output dim: 5, lower bound: -0.1638422, upper bound: 0.1660114
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.57
Output dim: 5, lower bound: -0.1660114, upper bound: 0.1638427
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.57
Output dim: 5, lower bound: -0.1638355, upper bound: 0.1660186

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4750376, 0.4734092
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4658456, 0.4671371
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4111414, 0.4125857
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3289227, 0.3284719
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3768857, 0.3770106
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3505986, 0.3499761
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3340774, 0.3338318
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4505699, 0.4505820
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2778757, 0.2770743
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4781709, 0.4773121

Time for backsubstitution: 22.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5857

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651793, upper bound: 0.1638353
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660175, upper bound: 0.1629968
time: 3.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4755123, 0.4729347
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4659915, 0.4669909
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4114273, 0.4122999
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3289119, 0.3284827
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3770211, 0.3768754
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3501649, 0.3504100
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3336601, 0.3342490
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4505942, 0.4505577
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2778347, 0.2771153
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4787898, 0.4766932

Time for backsubstitution: 21.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5857

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630035, upper bound: 0.1660112
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1638416, upper bound: 0.1651726
time: 3.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4729345, 0.4734526
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4658694, 0.4659915
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4111643, 0.4114273
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3284827, 0.3284799
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3768754, 0.3770111
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3504097, 0.3499806
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3340807, 0.3336601
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4505579, 0.4505820
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2771153, 0.2770901
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4766932, 0.4773436

Time for backsubstitution: 21.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5857

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651722, upper bound: 0.1638420
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660107, upper bound: 0.1630039
time: 3.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4734092, 0.4729781
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4660153, 0.4658453
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4114499, 0.4111414
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3284719, 0.3284907
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3770106, 0.3768759
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3499758, 0.3504143
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3336635, 0.3340774
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4505823, 0.4505572
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2770743, 0.2771311
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4773121, 0.4767246

Time for backsubstitution: 21.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5857

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5857

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1629963, upper bound: 0.1660180
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1638349, upper bound: 0.1651798
time: 3.15 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 27.87 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.87
Output dim: 5, lower bound: -0.1651793, upper bound: 0.1638353
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.87
Output dim: 5, lower bound: -0.1660175, upper bound: 0.1629968
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.87
Output dim: 5, lower bound: -0.1630035, upper bound: 0.1660112
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.87
Output dim: 5, lower bound: -0.1638416, upper bound: 0.1651726
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.87
Output dim: 5, lower bound: -0.1651722, upper bound: 0.1638420
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.87
Output dim: 5, lower bound: -0.1660107, upper bound: 0.1630039
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.87
Output dim: 5, lower bound: -0.1629963, upper bound: 0.1660180
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.87
Output dim: 5, lower bound: -0.1638349, upper bound: 0.1651798

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4741573, 0.4723461
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4661827, 0.4675441
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4083292, 0.4092107
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3264573, 0.3254284
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3738153, 0.3744519
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3466034, 0.3466460
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3310269, 0.3316755
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4394727, 0.4414907
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2756078, 0.2738504
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4802895, 0.4790652

Time for backsubstitution: 21.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2928

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2119

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1616729, upper bound: 0.1622330
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1634642, upper bound: 0.1601169
time: 3.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4739747, 0.4725287
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4662523, 0.4674747
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4077666, 0.4097733
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3258791, 0.3260064
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3743269, 0.3739402
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3472688, 0.3459808
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3319210, 0.3307815
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4414783, 0.4394851
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2746513, 0.2748065
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4799237, 0.4794309

Time for backsubstitution: 20.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2564

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1635680, upper bound: 0.1617019
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1647217, upper bound: 0.1605472
time: 3.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4746318, 0.4718716
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4663296, 0.4673979
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4086149, 0.4089251
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3264464, 0.3254392
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3739507, 0.3743165
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3461697, 0.3470799
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3306097, 0.3320928
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4394970, 0.4414661
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2755668, 0.2738910
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4809084, 0.4784462

Time for backsubstitution: 21.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 404

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1616964, upper bound: 0.1639227
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1615880, upper bound: 0.1648053
time: 3.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4744492, 0.4720542
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4663982, 0.4673285
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4080524, 0.4094877
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3258684, 0.3260173
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3744621, 0.3738048
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3468349, 0.3464144
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3315037, 0.3311987
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4415030, 0.4394605
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2746108, 0.2748475
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4805427, 0.4788120

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2119

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1601234, upper bound: 0.1634575
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1622393, upper bound: 0.1616655
time: 4.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4720542, 0.4723902
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4662066, 0.4663985
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4083524, 0.4080522
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3260173, 0.3254364
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3738050, 0.3744521
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3464146, 0.3466500
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3310302, 0.3315039
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4394603, 0.4414904
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2748473, 0.2738662
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4788117, 0.4790962

Time for backsubstitution: 21.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1641117, upper bound: 0.1599544
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1612841, upper bound: 0.1627815
time: 4.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4718716, 0.4725728
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4662757, 0.4663293
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4077897, 0.4086149
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3254392, 0.3260145
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3743165, 0.3739405
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3470798, 0.3459846
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3319243, 0.3306098
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4414663, 0.4394848
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2738912, 0.2748222
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4784465, 0.4794619

Time for backsubstitution: 21.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 404

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 212

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1655320, upper bound: 0.1626986
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1657060, upper bound: 0.1625244
time: 3.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4725289, 0.4719157
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4663525, 0.4662523
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4086382, 0.4077666
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3260064, 0.3254472
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3739402, 0.3743167
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3459806, 0.3470837
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3306130, 0.3319211
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4394851, 0.4414659
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2748063, 0.2739072
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4794307, 0.4784772

Time for backsubstitution: 21.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 1506

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 422

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1617754, upper bound: 0.1655815
time: 4.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1625621, upper bound: 0.1647949
time: 3.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4723463, 0.4720979
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4664221, 0.4661832
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4080756, 0.4083292
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3254284, 0.3260254
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3744519, 0.3738053
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3466461, 0.3464185
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3315071, 0.3310270
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4414907, 0.4394600
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2738502, 0.2748632
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4790654, 0.4788430

Time for backsubstitution: 21.57 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 422

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1626119, upper bound: 0.1647450
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1633987, upper bound: 0.1639583
time: 3.37 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.13
Output dim: 5, lower bound: -0.1616729, upper bound: 0.1622330
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.13
Output dim: 5, lower bound: -0.1634642, upper bound: 0.1601169
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.13
Output dim: 5, lower bound: -0.1635680, upper bound: 0.1617019
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.13
Output dim: 5, lower bound: -0.1647217, upper bound: 0.1605472
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.13
Output dim: 5, lower bound: -0.1616964, upper bound: 0.1639227
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.13
Output dim: 5, lower bound: -0.1615880, upper bound: 0.1648053
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.13
Output dim: 5, lower bound: -0.1601234, upper bound: 0.1634575
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.13
Output dim: 5, lower bound: -0.1622393, upper bound: 0.1616655
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.13
Output dim: 5, lower bound: -0.1641117, upper bound: 0.1599544
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.13
Output dim: 5, lower bound: -0.1612841, upper bound: 0.1627815
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.13
Output dim: 5, lower bound: -0.1655320, upper bound: 0.1626986
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.13
Output dim: 5, lower bound: -0.1657060, upper bound: 0.1625244
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.13
Output dim: 5, lower bound: -0.1617754, upper bound: 0.1655815
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.13
Output dim: 5, lower bound: -0.1625621, upper bound: 0.1647949
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.13
Output dim: 5, lower bound: -0.1626119, upper bound: 0.1647450
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.13
Output dim: 5, lower bound: -0.1633987, upper bound: 0.1639583

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4706626, 0.4689004
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4507382, 0.4540927
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4019156, 0.4059722
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3080159, 0.3129983
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3673856, 0.3665440
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3429916, 0.3409308
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3271707, 0.3272557
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4382370, 0.4355240
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2704268, 0.2729945
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4761939, 0.4743021

Time for backsubstitution: 22.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2523

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1506

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1611233, upper bound: 0.1604907
time: 4.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1646651, upper bound: 0.1569476
time: 3.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4734156, 0.4705162
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4651384, 0.4658060
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4068704, 0.4074461
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3196206, 0.3195744
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3690999, 0.3712349
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3413498, 0.3424937
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3238434, 0.3247757
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4386117, 0.4410484
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2739973, 0.2726097
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4794445, 0.4767022

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 422

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1506

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1579896, upper bound: 0.1647477
time: 4.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1615305, upper bound: 0.1612093
time: 4.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4718430, 0.4725275
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4640150, 0.4637477
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4079199, 0.4087505
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3220184, 0.3230071
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3754120, 0.3754869
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3466024, 0.3455062
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3323585, 0.3316271
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4417071, 0.4393687
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2733834, 0.2748029
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4784489, 0.4794691

Time for backsubstitution: 21.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2570

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 80

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1644714, upper bound: 0.1588107
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1616460, upper bound: 0.1616381
time: 3.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4718268, 0.4725435
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4636917, 0.4640682
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4079251, 0.4087451
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3224320, 0.3225889
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3758605, 0.3750358
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3465986, 0.3455074
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3329388, 0.3310437
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4413495, 0.4397261
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2738719, 0.2743130
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4784532, 0.4794638

Time for backsubstitution: 22.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2119

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1621995, upper bound: 0.1609221
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1639909, upper bound: 0.1588060
time: 2.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4708164, 0.4703662
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4665165, 0.4664309
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4083090, 0.4057164
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3222579, 0.3220253
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3710694, 0.3707757
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3452528, 0.3463989
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3308156, 0.3317940
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4379969, 0.4405360
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2636733, 0.2610023
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4755125, 0.4756758

Time for backsubstitution: 21.90 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.50 + 550.67 = 609.16 seconds
