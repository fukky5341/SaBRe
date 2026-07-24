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
execution time: IAR + RelationalAnalysis = 21.68 + 33.36 = 55.05 seconds
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

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 6165

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660230, upper bound: 0.1660166
time: 4.41 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660162, upper bound: 0.1660234
time: 3.03 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.60 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.60
Output dim: 5, lower bound: -0.1660230, upper bound: 0.1660166
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.60
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

Time for backsubstitution: 20.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 836

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 5857

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651842, upper bound: 0.1660161
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660224, upper bound: 0.1651775
time: 4.32 seconds

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

Time for backsubstitution: 20.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 836

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 5857

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651771, upper bound: 0.1660222
time: 4.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660156, upper bound: 0.1651847
time: 3.29 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.69 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.69
Output dim: 5, lower bound: -0.1651842, upper bound: 0.1660161
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.69
Output dim: 5, lower bound: -0.1660224, upper bound: 0.1651775
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.69
Output dim: 5, lower bound: -0.1651771, upper bound: 0.1660222
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.69
Output dim: 5, lower bound: -0.1660156, upper bound: 0.1651847

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4769988, 0.4747133
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4669199, 0.4681349
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4100430, 0.4106388
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3264062, 0.3253881
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3733151, 0.3738160
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3468158, 0.3472919
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3331158, 0.3341815
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4396436, 0.4416375
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2757839, 0.2740674
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4839959, 0.4821529

Time for backsubstitution: 21.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 836

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 836

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651793, upper bound: 0.1638353
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1630035, upper bound: 0.1660112
time: 3.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4768162, 0.4748960
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4669895, 0.4680655
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4094803, 0.4112015
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3258281, 0.3259662
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3738265, 0.3733046
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3474810, 0.3466268
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3340099, 0.3332875
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4416497, 0.4396317
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2748278, 0.2750237
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4836307, 0.4825182

Time for backsubstitution: 21.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 836

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 836

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660175, upper bound: 0.1629968
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1638416, upper bound: 0.1651726
time: 3.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4748960, 0.4747574
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4669442, 0.4669893
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4100659, 0.4094803
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3259662, 0.3253961
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3733046, 0.3738160
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3466268, 0.3472958
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3331194, 0.3340099
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4396317, 0.4416378
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2750236, 0.2740839
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4825182, 0.4821835

Time for backsubstitution: 21.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 836

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 836

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1651722, upper bound: 0.1638420
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1629963, upper bound: 0.1660180
time: 3.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4747133, 0.4749401
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4670134, 0.4669201
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.4095032, 0.4100430
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3253881, 0.3259741
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3738163, 0.3733044
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3472922, 0.3466303
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3340135, 0.3331158
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4416373, 0.4396319
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2740675, 0.2750399
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4821529, 0.4825492

Time for backsubstitution: 21.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 836

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 836

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1660107, upper bound: 0.1630039
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1638349, upper bound: 0.1651798
time: 3.25 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.63 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.63
Output dim: 5, lower bound: -0.1651793, upper bound: 0.1638353
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.63
Output dim: 5, lower bound: -0.1630035, upper bound: 0.1660112
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.63
Output dim: 5, lower bound: -0.1660175, upper bound: 0.1629968
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.63
Output dim: 5, lower bound: -0.1638416, upper bound: 0.1651726
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.63
Output dim: 5, lower bound: -0.1651722, upper bound: 0.1638420
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.63
Output dim: 5, lower bound: -0.1629963, upper bound: 0.1660180
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.63
Output dim: 5, lower bound: -0.1660107, upper bound: 0.1630039
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.63
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

Time for backsubstitution: 21.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1506

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 3, pos: 2481

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1640292, upper bound: 0.1591966
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1605354, upper bound: 0.1626852
time: 3.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 21.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1506

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 3, pos: 2481

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1618533, upper bound: 0.1613725
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1583595, upper bound: 0.1648611
time: 3.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 20.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1506

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 3, pos: 2481

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1648673, upper bound: 0.1583525
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1613794, upper bound: 0.1618467
time: 3.12 seconds

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

Time for backsubstitution: 21.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1506

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 3, pos: 2481

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1626915, upper bound: 0.1605284
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1592035, upper bound: 0.1640226
time: 3.14 seconds

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

Time for backsubstitution: 21.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1506

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 3, pos: 2481

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1640222, upper bound: 0.1592039
time: 3.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1605280, upper bound: 0.1626919
time: 3.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 21.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1506

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 2481

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1618463, upper bound: 0.1613798
time: 5.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1583522, upper bound: 0.1648678
time: 3.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 22.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1506

Time for candidate selection: 0.39 seconds

### Candidate
type: DSZ, layer: 3, pos: 2481

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1648606, upper bound: 0.1583595
time: 5.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1613720, upper bound: 0.1618532
time: 5.08 seconds

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

Time for backsubstitution: 22.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1506

Time for candidate selection: 0.39 seconds

### Candidate
type: DSZ, layer: 3, pos: 2481

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1626849, upper bound: 0.1605357
time: 4.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1591962, upper bound: 0.1640292
time: 5.70 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 32.94 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.94
Output dim: 5, lower bound: -0.1640292, upper bound: 0.1591966
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.94
Output dim: 5, lower bound: -0.1605354, upper bound: 0.1626852
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.94
Output dim: 5, lower bound: -0.1618533, upper bound: 0.1613725
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.94
Output dim: 5, lower bound: -0.1583595, upper bound: 0.1648611
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.94
Output dim: 5, lower bound: -0.1648673, upper bound: 0.1583525
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.94
Output dim: 5, lower bound: -0.1613794, upper bound: 0.1618467
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.94
Output dim: 5, lower bound: -0.1626915, upper bound: 0.1605284
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.94
Output dim: 5, lower bound: -0.1592035, upper bound: 0.1640226
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.94
Output dim: 5, lower bound: -0.1640222, upper bound: 0.1592039
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.94
Output dim: 5, lower bound: -0.1605280, upper bound: 0.1626919
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.94
Output dim: 5, lower bound: -0.1618463, upper bound: 0.1613798
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.94
Output dim: 5, lower bound: -0.1583522, upper bound: 0.1648678
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.94
Output dim: 5, lower bound: -0.1648606, upper bound: 0.1583595
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.94
Output dim: 5, lower bound: -0.1613720, upper bound: 0.1618532
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.94
Output dim: 5, lower bound: -0.1626849, upper bound: 0.1605357
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.94
Output dim: 5, lower bound: -0.1591962, upper bound: 0.1640292

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4678268, 0.4649222
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4316597, 0.4384298
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.3741176, 0.3736432
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3121350, 0.3125727
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3680849, 0.3682933
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3374453, 0.3391423
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3283646, 0.3281732
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.3971617, 0.4042058
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2713263, 0.2694716
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4621572, 0.4618249

Time for backsubstitution: 21.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1506

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2523

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1571538, upper bound: 0.1641246
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1576489, upper bound: 0.1635336
time: 3.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4670250, 0.4657238
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4372845, 0.4328055
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.3724847, 0.3752761
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3130127, 0.3116950
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3683038, 0.3680744
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3393314, 0.3372564
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3280015, 0.3285363
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4042175, 0.3971496
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2702320, 0.2705660
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4633026, 0.4606795

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1506

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2523

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1635383, upper bound: 0.1576421
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1641311, upper bound: 0.1571488
time: 5.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4657238, 0.4649656
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4316835, 0.4372845
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.3741403, 0.3724847
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3116950, 0.3125811
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3680744, 0.3682935
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3372564, 0.3391466
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3283679, 0.3280015
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.3971493, 0.4042058
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2705662, 0.2694874
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4606795, 0.4618554

Time for backsubstitution: 22.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1506

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 2523

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1571486, upper bound: 0.1641316
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1576417, upper bound: 0.1635388
time: 3.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.1680317, -8.3224335, -9.1680317, -8.3224335, -0.4649222, 0.4657671
1: -10.3162136, -9.4436111, -10.3162136, -9.4436111, -0.4373078, 0.4316599
2: -8.5261040, -7.8132076, -8.5261040, -7.8132076, -0.3725073, 0.3741176
3: -10.0638485, -9.3873940, -10.0638485, -9.3873940, -0.3125727, 0.3117033
4: -9.9685507, -9.3377151, -9.9685507, -9.3377151, -0.3682933, 0.3680749
5: 7.7518911, 8.3284512, 7.7518911, 8.3284512, -0.3391423, 0.3372607
6: -4.2394090, -3.5505164, -4.2394090, -3.5505164, -0.3280048, 0.3283646
7: -13.7619133, -12.8516111, -13.7619133, -12.8516111, -0.4042056, 0.3971496
8: 0.9233379, 1.3804231, 0.9233379, 1.3804231, -0.2694714, 0.2705820
9: -6.6247835, -6.0456238, -6.6247835, -6.0456238, -0.4618249, 0.4607100

Time for backsubstitution: 21.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2523
type: DSZ, layer: 3, pos: 80
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2119
type: DSZ, layer: 3, pos: 2564
type: DSZ, layer: 3, pos: 212
type: DSZ, layer: 3, pos: 2928
type: DSZ, layer: 3, pos: 37
type: DSZ, layer: 3, pos: 404
type: DSZ, layer: 3, pos: 2570
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1506

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 2523

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1635332, upper bound: 0.1576489
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1641242, upper bound: 0.1571540
time: 3.12 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 28.64 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.64
Output dim: 5, lower bound: -0.1571538, upper bound: 0.1641246
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.64
Output dim: 5, lower bound: -0.1576489, upper bound: 0.1635336
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.64
Output dim: 5, lower bound: -0.1635383, upper bound: 0.1576421
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.64
Output dim: 5, lower bound: -0.1641311, upper bound: 0.1571488
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.64
Output dim: 5, lower bound: -0.1571486, upper bound: 0.1641316
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.64
Output dim: 5, lower bound: -0.1576417, upper bound: 0.1635388
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.64
Output dim: 5, lower bound: -0.1635332, upper bound: 0.1576489
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.64
Output dim: 5, lower bound: -0.1641242, upper bound: 0.1571540

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 55.05 + 536.67 = 591.72 seconds
