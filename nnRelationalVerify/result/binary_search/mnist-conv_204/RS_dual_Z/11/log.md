## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.41320994132
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.6942806, -5.0616770, -9.6942806, -5.0616770, -4.3366542, 4.3366537)
1: (-15.0952425, -10.8431473, -15.0952425, -10.8431473, -4.2520952, 4.2520952)
2: (-9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.2964392, 3.2964392)
3: (-11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.1194048, 4.1194048)
4: (-5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.5223343, 3.5223343)
5: (-3.5736499, -0.4953117, -3.5736499, -0.4953117, -3.0783381, 3.0783381)
6: (-11.5837259, -6.9704914, -11.5837259, -6.9704914, -4.5754027, 4.5754023)
7: (-2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6390057, 3.6390057)
8: (-5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.6043172, 3.6043172)
9: (0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.6205788, 2.6205788)

## BASE Result
execution time: IAR + LP analysis = 15.26 + 33.76 = 49.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -2.1425934, upper bound: 2.1425903


# Binary Search by BASE starts (time budget: 3550.98 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.567917585372925
rel_dist={9: [-1.824091324888559, 1.824091281312575]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.44260835647583
rel_dist={9: [-1.592849651897651, 1.5928489343896768]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.3590686321258545
rel_dist={9: [-1.414766908177059, 1.4147664541724545]}

## Binary Search Result
Binary search time: 152.46 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 3398.52 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 485

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240801, upper bound: 1.8131312
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8131314, upper bound: 1.8240801
time: 4.64 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.54 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.54
Output dim: 9, lower bound: -1.8240801, upper bound: 1.8131312
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.54
Output dim: 9, lower bound: -1.8131314, upper bound: 1.8240801

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6230783, 3.6133938
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7465677, 3.7384958
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0199718, 3.0139136
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0610595, 4.0585113
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4221153, 3.4140148
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9009066, 2.9078069
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.9386945, 3.9350290
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6055794, 3.6013894
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.2107663, 3.2155864
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5522890, 2.5470889

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6238

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240766, upper bound: 1.8102313
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8212446, upper bound: 1.8131275
time: 5.55 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6133928, 3.6230783
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7384958, 3.7465677
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0139141, 3.0199718
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0585103, 4.0610595
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4140148, 3.4221148
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9078074, 2.9009070
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.9350286, 3.9386954
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6013899, 3.6055794
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.2155862, 3.2107663
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5470886, 2.5522897

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6238

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8131278, upper bound: 1.8212446
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8102315, upper bound: 1.8240765
time: 4.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.38 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.38
Output dim: 9, lower bound: -1.8240766, upper bound: 1.8102313
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.38
Output dim: 9, lower bound: -1.8212446, upper bound: 1.8131275
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.38
Output dim: 9, lower bound: -1.8131278, upper bound: 1.8212446
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.38
Output dim: 9, lower bound: -1.8102315, upper bound: 1.8240765

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6013222, 3.6001115
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7480984, 3.7384806
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0000587, 3.0016155
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0695791, 4.0645909
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4209528, 3.4111490
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8923702, 2.9013815
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8953872, 3.9027686
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6068225, 3.6013708
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.2114811, 3.2148004
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5453556, 2.5375443

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8162297, upper bound: 1.8102286
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240689, upper bound: 1.8024504
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6097965, 3.5916376
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7465534, 3.7400265
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0076747, 2.9940000
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0671396, 4.0670304
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4192486, 3.4128532
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8944817, 2.8992701
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.9064345, 3.8917217
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6055608, 3.6026325
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.2099810, 3.2163007
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5427454, 2.5401545

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8133947, upper bound: 1.8131248
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8212414, upper bound: 1.8053095
time: 5.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5916386, 3.6097963
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7400265, 3.7465525
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9940009, 3.0076737
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0670309, 4.0671391
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4128523, 3.4192491
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8992710, 2.8944817
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8917212, 3.9064345
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6026320, 3.6055608
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.2163010, 3.2099805
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5401542, 2.5427451

Time for backsubstitution: 17.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053091, upper bound: 1.8212409
time: 7.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8131247, upper bound: 1.8133947
time: 4.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6001110, 3.6013222
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7384815, 3.7480984
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0016150, 3.0000582
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0645905, 4.0695796
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4111490, 3.4209533
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9013824, 2.8923702
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.9027686, 3.8953881
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6013713, 3.6068225
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.2148008, 3.2114809
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5375440, 2.5453553

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8024503, upper bound: 1.8240688
time: 8.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8102284, upper bound: 1.8162298
time: 4.66 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 28.46 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.46
Output dim: 9, lower bound: -1.8162297, upper bound: 1.8102286
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.46
Output dim: 9, lower bound: -1.8240689, upper bound: 1.8024504
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.46
Output dim: 9, lower bound: -1.8133947, upper bound: 1.8131248
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.46
Output dim: 9, lower bound: -1.8212414, upper bound: 1.8053095
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.46
Output dim: 9, lower bound: -1.8053091, upper bound: 1.8212409
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.46
Output dim: 9, lower bound: -1.8131247, upper bound: 1.8133947
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 28.46
Output dim: 9, lower bound: -1.8024503, upper bound: 1.8240688
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 28.46
Output dim: 9, lower bound: -1.8102284, upper bound: 1.8162298

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5497375, 3.5313966
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7500949, 3.7384758
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9743834, 2.9850421
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0757828, 4.0814581
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4170222, 3.4009452
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8946457, 2.9009008
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8711367, 3.8675714
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6111789, 3.5985918
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1673803, 3.1835673
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5534406, 2.5439570

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8074167, upper bound: 1.8102225
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8162235, upper bound: 1.8015441
time: 4.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5326076, 3.5485265
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7480941, 3.7404766
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9834852, 2.9759402
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0864477, 4.0707932
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4107490, 3.4072189
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8918896, 2.9036560
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8601904, 3.8785172
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6040435, 3.6057267
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1802473, 3.1707001
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5517678, 2.5456297

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8152473, upper bound: 1.8024445
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240628, upper bound: 1.7937041
time: 4.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5582108, 3.5229228
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7485480, 3.7400217
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9819994, 2.9774265
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0733423, 4.0838985
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4153190, 3.4026494
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8967571, 2.8987894
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8821831, 3.8565245
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6099162, 3.5998535
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1658792, 3.1850677
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5508304, 2.5465672

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8046023, upper bound: 1.8131188
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8133885, upper bound: 1.8044101
time: 4.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5410819, 3.5400527
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7465472, 3.7420225
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9911003, 2.9683251
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0840082, 4.0732336
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4090447, 3.4089231
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8940010, 2.9015446
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8712368, 3.8674703
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6027827, 3.6069884
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1787481, 3.1722004
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5491576, 2.5482399

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8124017, upper bound: 1.8053027
time: 7.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8212352, upper bound: 1.7965691
time: 4.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5400529, 3.5410814
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7420230, 3.7465477
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9683256, 2.9911003
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0732327, 4.0840082
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4089227, 3.4090452
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9015446, 2.8940010
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8674707, 3.8712373
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6069884, 3.6027818
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1722002, 3.1787474
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5482402, 2.5491579

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7965692, upper bound: 1.8212350
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053029, upper bound: 1.8124016
time: 4.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5229230, 3.5582113
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7400222, 3.7485485
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9774265, 2.9819989
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0838985, 4.0733423
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4026484, 3.4153190
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8987904, 2.8967566
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8565245, 3.8821836
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5998549, 3.6099162
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1850672, 3.1658802
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5465674, 2.5508306

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8044098, upper bound: 1.8133888
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8131186, upper bound: 1.8046023
time: 4.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5485272, 3.5326076
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7404761, 3.7480936
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9759398, 2.9834847
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0707932, 4.0864477
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4072185, 3.4107494
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9036560, 2.8918896
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8785162, 3.8601909
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6057277, 3.6040435
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1706991, 3.1802478
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5456300, 2.5517681

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7937043, upper bound: 1.8240632
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8024442, upper bound: 1.8152472
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5313973, 3.5497375
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7384753, 3.7500944
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9850416, 2.9743834
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0814590, 4.0757818
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4009452, 3.4170232
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9009018, 2.8946452
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8675709, 3.8711367
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5985923, 3.6111779
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1835680, 3.1673806
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5439572, 2.5534408

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8015439, upper bound: 1.8162240
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8102223, upper bound: 1.8074167
time: 4.78 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.69 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.8074167, upper bound: 1.8102225
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.8162235, upper bound: 1.8015441
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.8152473, upper bound: 1.8024445
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.8240628, upper bound: 1.7937041
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.8046023, upper bound: 1.8131188
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.8133885, upper bound: 1.8044101
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.8124017, upper bound: 1.8053027
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.8212352, upper bound: 1.7965691
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.7965692, upper bound: 1.8212350
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.8053029, upper bound: 1.8124016
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.8044098, upper bound: 1.8133888
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.8131186, upper bound: 1.8046023
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.7937043, upper bound: 1.8240632
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.8024442, upper bound: 1.8152472
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.8015439, upper bound: 1.8162240
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.69
Output dim: 9, lower bound: -1.8102223, upper bound: 1.8074167

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.4280519, 3.3691511
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7577071, 3.7487659
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8923998, 2.8757527
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0471001, 4.0432305
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4362164, 3.4161463
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8884258, 2.8962321
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.7334023, 3.6839280
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6212053, 3.6065302
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1393385, 3.1462018
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5526648, 2.5433753

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8074148, upper bound: 1.8090428
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8062565, upper bound: 1.8102205
time: 4.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.3874922, 3.4097111
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7603850, 3.7460880
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8650942, 2.9030583
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0375538, 4.0527768
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4322243, 3.4201388
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8899765, 2.8946815
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6874924, 3.7298398
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6191158, 3.6086202
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1300144, 3.1555254
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5528593, 2.5431812

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8162216, upper bound: 1.8003607
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8150261, upper bound: 1.8015419
time: 5.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.4109221, 3.3862810
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7557063, 3.7507672
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9015017, 2.8666508
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0577660, 4.0325646
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4299421, 3.4224200
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8856697, 2.8989873
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.7224579, 3.6948738
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6140709, 3.6136651
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1522055, 3.1333346
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5509920, 2.5450480

Time for backsubstitution: 14.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8152455, upper bound: 1.8012168
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8141787, upper bound: 1.8024424
time: 4.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.3703623, 3.4268410
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7583842, 3.7480888
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8741961, 2.8939569
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0482197, 4.0421109
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4259501, 3.4264126
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8872204, 2.8974366
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6765480, 3.7407861
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6119814, 3.6157551
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1428823, 3.1426582
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5511866, 2.5448539

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240610, upper bound: 1.7924921
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8229283, upper bound: 1.7937025
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.4365253, 3.3606772
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7561603, 3.7503123
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9000149, 2.8681371
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0446606, 4.0456700
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4345121, 3.4178505
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8905373, 2.8941207
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.7444515, 3.6728811
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6199446, 3.6077919
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1378384, 3.1477022
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5500546, 2.5459855

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8046004, upper bound: 1.8119372
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8034392, upper bound: 1.8131167
time: 4.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.3959656, 3.4012372
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7588382, 3.7476339
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8727093, 2.8954427
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0351143, 4.0552163
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4305201, 3.4218426
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8920879, 2.8925700
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6985397, 3.7187910
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6178541, 3.6098819
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1285143, 3.1570258
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5502491, 2.5457914

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8133866, upper bound: 1.8032267
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8121774, upper bound: 1.8044079
time: 4.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.4193964, 3.3778071
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7541595, 3.7523131
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9091167, 2.8590357
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0553265, 4.0350041
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4282389, 3.4241242
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8877811, 2.8968759
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.7335052, 3.6838269
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6128092, 3.6149268
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1507053, 3.1348350
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5483818, 2.5476582

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8123999, upper bound: 1.8040723
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8113213, upper bound: 1.8053011
time: 4.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.3788357, 3.4183669
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7568374, 3.7496347
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8818111, 2.8863413
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0457802, 4.0445514
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4242468, 3.4281163
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8893318, 2.8953252
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6875935, 3.7297368
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6107197, 3.6170168
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1413822, 3.1441586
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5485764, 2.5474641

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8212333, upper bound: 1.7953570
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8200660, upper bound: 1.7965677
time: 8.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.4183664, 3.3788359
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7496352, 3.7568378
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8863411, 2.8818109
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0445518, 4.0457797
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4281168, 3.4242463
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8953247, 2.8893323
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.7297363, 3.6875939
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6170168, 3.6107202
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1441584, 3.1413820
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5474644, 2.5485761

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7965671, upper bound: 1.8200658
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7953567, upper bound: 1.8212334
time: 4.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.3778067, 3.4193959
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7523131, 3.7541599
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8590355, 2.9091165
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0350046, 4.0553260
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4241247, 3.4282389
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8968754, 2.8877816
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6838264, 3.7335062
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6149263, 3.6128101
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1348343, 3.1507053
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5476580, 2.5483820

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8053009, upper bound: 1.8113211
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8040723, upper bound: 1.8124001
time: 4.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.4012375, 3.3959656
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7476344, 3.7588387
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8954430, 2.8727095
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0552168, 4.0351138
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4218426, 3.4305201
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8925705, 2.8920879
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.7187901, 3.6985402
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6098814, 3.6178546
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1570263, 3.1285148
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5457916, 2.5502489

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8044077, upper bound: 1.8121777
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8032263, upper bound: 1.8133869
time: 4.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.3606768, 3.4365256
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7503123, 3.7561607
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8681374, 2.9000151
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0456696, 4.0446610
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4178505, 3.4345121
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8941212, 2.8905373
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6728802, 3.7444520
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6077909, 3.6199446
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1477022, 3.1378384
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5459852, 2.5500548

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8131166, upper bound: 1.8034392
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8119371, upper bound: 1.8046002
time: 4.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.4268408, 3.3703618
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7480884, 3.7583842
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8939571, 2.8741953
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0421104, 4.0482192
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4264126, 3.4259505
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8974361, 2.8872209
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.7407856, 3.6765475
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6157541, 3.6119819
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1426582, 3.1428823
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5448542, 2.5511863

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7937023, upper bound: 1.8229286
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7924922, upper bound: 1.8240610
time: 4.19 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.3862810, 3.4109221
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7507682, 3.7557058
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8666515, 2.9015009
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0325642, 4.0577664
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4224195, 3.4299426
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8989868, 2.8856702
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6948738, 3.7224574
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6136646, 3.6140718
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1333342, 3.1522059
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5450478, 2.5509923

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8024421, upper bound: 1.8141788
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8012164, upper bound: 1.8152458
time: 4.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.4097109, 3.3874917
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7460876, 3.7603850
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9030590, 2.8650939
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0527763, 4.0375543
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4201384, 3.4322243
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8946819, 2.8899765
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.7298393, 3.6874933
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6086197, 3.6191163
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1555252, 3.1300151
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5431814, 2.5528591

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8015418, upper bound: 1.8150263
time: 8.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8003604, upper bound: 1.8162218
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.3691511, 3.4280517
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7487655, 3.7577066
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8757524, 2.8923995
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0432301, 4.0471005
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4161463, 3.4362164
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.8962326, 2.8884258
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6839275, 3.7334032
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6065302, 3.6212063
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1462021, 3.1393387
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5433750, 2.5526650

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8102203, upper bound: 1.8062565
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8090428, upper bound: 1.8074147
time: 4.50 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8074148, upper bound: 1.8090428
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8062565, upper bound: 1.8102205
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8162216, upper bound: 1.8003607
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8150261, upper bound: 1.8015419
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8152455, upper bound: 1.8012168
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8141787, upper bound: 1.8024424
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8240610, upper bound: 1.7924921
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8229283, upper bound: 1.7937025
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8046004, upper bound: 1.8119372
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8034392, upper bound: 1.8131167
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8133866, upper bound: 1.8032267
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8121774, upper bound: 1.8044079
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8123999, upper bound: 1.8040723
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8113213, upper bound: 1.8053011
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8212333, upper bound: 1.7953570
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8200660, upper bound: 1.7965677
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.7965671, upper bound: 1.8200658
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.7953567, upper bound: 1.8212334
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8053009, upper bound: 1.8113211
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8040723, upper bound: 1.8124001
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8044077, upper bound: 1.8121777
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8032263, upper bound: 1.8133869
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8131166, upper bound: 1.8034392
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8119371, upper bound: 1.8046002
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.7937023, upper bound: 1.8229286
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.7924922, upper bound: 1.8240610
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8024421, upper bound: 1.8141788
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8012164, upper bound: 1.8152458
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8015418, upper bound: 1.8150263
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8003604, upper bound: 1.8162218
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8102203, upper bound: 1.8062565
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.96
Output dim: 9, lower bound: -1.8090428, upper bound: 1.8074147
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.567917585372925
rel_dist={9: [-1.824091324888559, 1.824091281312575]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 485

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928443, upper bound: 1.5874001
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5874001, upper bound: 1.5928443
time: 4.67 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.81
Output dim: 9, lower bound: -1.5928443, upper bound: 1.5874001
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.81
Output dim: 9, lower bound: -1.5874001, upper bound: 1.5928443

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2759581, 3.2711163
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4109058, 3.4068704
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8274717, 2.8244424
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7778592, 3.7765841
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2522144, 3.2481656
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6768427, 2.6802912
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6094308, 3.6075983
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4373198, 3.4352250
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9356284, 2.9380386
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4243793, 2.4217794

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6238

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928427, upper bound: 1.5857149
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5911661, upper bound: 1.5873976
time: 4.82 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2711153, 3.2759585
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4068699, 3.4109063
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8244429, 2.8274717
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7765841, 3.7778602
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2481651, 3.2522154
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6802912, 2.6768417
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6075978, 3.6094313
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4352255, 3.4373198
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9380383, 2.9356287
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4217787, 2.4243796

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6238

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5873984, upper bound: 1.5911668
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5857161, upper bound: 1.5928427
time: 4.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.85 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.85
Output dim: 9, lower bound: -1.5928427, upper bound: 1.5857149
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.85
Output dim: 9, lower bound: -1.5911661, upper bound: 1.5873976
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.85
Output dim: 9, lower bound: -1.5873984, upper bound: 1.5911668
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.85
Output dim: 9, lower bound: -1.5857161, upper bound: 1.5928427

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2542019, 3.2535970
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4116650, 3.4068551
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8075576, 2.8083365
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7851591, 3.7826638
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2502012, 3.2452998
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6683044, 2.6728101
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5661235, 3.5698142
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4379325, 3.4352064
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9355927, 2.9372525
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4161403, 2.4122348

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5883008, upper bound: 1.5857104
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928370, upper bound: 1.5811773
time: 4.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2584400, 3.2493601
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4108915, 3.4076285
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8113666, 2.8045287
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7839394, 3.7838845
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2493486, 3.2461514
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6693611, 2.6717544
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5716472, 3.5642910
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4373012, 3.4358373
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9348421, 2.9380028
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4148347, 2.4135399

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5866244, upper bound: 1.5873934
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5911604, upper bound: 1.5828573
time: 4.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2493610, 3.2584393
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4076290, 3.4108911
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8045287, 2.8113658
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7838850, 3.7839398
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2461510, 3.2493496
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6717548, 2.6693606
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5642905, 3.5716476
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4358373, 3.4373012
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9380026, 2.9348426
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4135396, 2.4148350

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5828585, upper bound: 1.5911605
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5873926, upper bound: 1.5866244
time: 4.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2535973, 3.2542024
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4068556, 3.4116645
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8083367, 2.8075581
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7826643, 3.7851586
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2452993, 3.2502017
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6728096, 2.6683049
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5698142, 3.5661240
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4352069, 3.4379320
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9372530, 2.9355927
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4122350, 2.4161406

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5811784, upper bound: 1.5928369
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5857104, upper bound: 1.5883007
time: 4.47 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.79 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.79
Output dim: 9, lower bound: -1.5883008, upper bound: 1.5857104
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.79
Output dim: 9, lower bound: -1.5928370, upper bound: 1.5811773
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.79
Output dim: 9, lower bound: -1.5866244, upper bound: 1.5873934
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.79
Output dim: 9, lower bound: -1.5911604, upper bound: 1.5828573
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.79
Output dim: 9, lower bound: -1.5828585, upper bound: 1.5911605
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.79
Output dim: 9, lower bound: -1.5873926, upper bound: 1.5866244
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.79
Output dim: 9, lower bound: -1.5811784, upper bound: 1.5928369
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.79
Output dim: 9, lower bound: -1.5857104, upper bound: 1.5883007

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1940522, 3.1848822
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4126601, 3.4068503
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7818832, 2.7872124
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7913618, 3.7941999
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2431340, 3.2350960
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6692028, 2.6723294
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5363989, 3.5346169
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4387221, 3.4324274
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8914919, 2.8995857
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4233890, 2.4186475

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838854, upper bound: 1.5857076
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5882978, upper bound: 1.5812907
time: 4.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1854882, 3.1934471
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4116597, 3.4078512
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7864332, 2.7826614
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7966948, 3.7888670
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2399974, 3.2382326
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6678238, 2.6737075
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5309267, 3.5400901
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4351535, 3.4359946
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8979254, 2.8931522
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4225535, 2.4194839

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5884236, upper bound: 1.5811745
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928340, upper bound: 1.5767516
time: 4.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1982894, 3.1806452
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4118876, 3.4076238
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7856913, 2.7834046
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7901421, 3.7954197
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2422824, 3.2359476
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6702576, 2.6712737
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5419226, 3.5290937
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4380908, 3.4330583
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8907423, 2.9003360
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4220843, 2.4199526

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5822049, upper bound: 1.5873905
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5866214, upper bound: 1.5829713
time: 4.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1897244, 3.1892102
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4108863, 3.4086242
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7902412, 2.7788539
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7954750, 3.7900877
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2391448, 3.2390847
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6688805, 2.6726513
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5364504, 3.5345669
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4345222, 3.4366255
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8971758, 2.8939025
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4212480, 2.4207890

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5867428, upper bound: 1.5828551
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5911574, upper bound: 1.5784320
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1892104, 3.1897244
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4086242, 3.4108863
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7788534, 2.7902415
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7900867, 3.7954741
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2390838, 3.2391458
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6726513, 2.6688800
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5345659, 3.5364504
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4366260, 3.4345222
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8939028, 2.8971758
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4207892, 2.4212477

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5784332, upper bound: 1.5911574
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5828554, upper bound: 1.5867427
time: 4.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1806455, 3.1982894
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4076238, 3.4118872
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7834044, 2.7856908
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7954197, 3.7901421
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2359471, 3.2422824
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6712742, 2.6702576
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5290937, 3.5419230
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4330592, 3.4380898
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9003363, 2.8907423
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4199529, 2.4220841

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5829726, upper bound: 1.5866216
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5873896, upper bound: 1.5822062
time: 4.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1934476, 3.1854875
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4078517, 3.4116597
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7826614, 2.7864337
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7888670, 3.7966948
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2382321, 3.2399979
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6737080, 2.6678243
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5400896, 3.5309267
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4359946, 3.4351530
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8931513, 2.8979261
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4194837, 2.4225533

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5767525, upper bound: 1.5928340
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5811754, upper bound: 1.5884236
time: 4.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1848826, 3.1940525
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4068503, 3.4126601
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7872124, 2.7818830
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7941999, 3.7913618
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2350955, 3.2431345
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6723289, 2.6692019
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5346165, 3.5363998
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4324279, 3.4387207
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8995857, 2.8914924
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4186473, 2.4233892

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5812908, upper bound: 1.5882977
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5857074, upper bound: 1.5838868
time: 4.72 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.03
Output dim: 9, lower bound: -1.5838854, upper bound: 1.5857076
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.03
Output dim: 9, lower bound: -1.5882978, upper bound: 1.5812907
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.03
Output dim: 9, lower bound: -1.5884236, upper bound: 1.5811745
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.03
Output dim: 9, lower bound: -1.5928340, upper bound: 1.5767516
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.03
Output dim: 9, lower bound: -1.5822049, upper bound: 1.5873905
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.03
Output dim: 9, lower bound: -1.5866214, upper bound: 1.5829713
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.03
Output dim: 9, lower bound: -1.5867428, upper bound: 1.5828551
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.03
Output dim: 9, lower bound: -1.5911574, upper bound: 1.5784320
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.03
Output dim: 9, lower bound: -1.5784332, upper bound: 1.5911574
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.03
Output dim: 9, lower bound: -1.5828554, upper bound: 1.5867427
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.03
Output dim: 9, lower bound: -1.5829726, upper bound: 1.5866216
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.03
Output dim: 9, lower bound: -1.5873896, upper bound: 1.5822062
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.03
Output dim: 9, lower bound: -1.5767525, upper bound: 1.5928340
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.03
Output dim: 9, lower bound: -1.5811754, upper bound: 1.5884236
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.03
Output dim: 9, lower bound: -1.5812908, upper bound: 1.5882977
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.03
Output dim: 9, lower bound: -1.5857074, upper bound: 1.5838868

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0520868, 3.0226367
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4202724, 3.4158020
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6862469, 2.6779230
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7579060, 3.7559714
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2603312, 3.2502971
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6629829, 2.6668854
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3757114, 3.3509736
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4477034, 3.4403658
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8587885, 2.8622203
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4226141, 2.4179685

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838842, upper bound: 1.5849099
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5830993, upper bound: 1.5857052
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0318069, 3.0429168
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4216113, 3.4144626
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6725941, 2.6915758
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7531338, 3.7607446
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2583361, 3.2522931
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6637573, 2.6661100
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3527565, 3.3739300
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4466581, 3.4414110
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8541269, 2.8668823
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4227104, 2.4178717

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5882966, upper bound: 1.5804942
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5875011, upper bound: 1.5812886
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0435219, 3.0312016
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4192729, 3.4168024
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6907969, 2.6733720
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7632389, 3.7506385
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2571955, 3.2534337
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6616049, 2.6682634
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3702374, 3.3564467
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4441366, 3.4439330
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8652220, 2.8557868
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4217777, 2.4188049

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5884224, upper bound: 1.5803733
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5876384, upper bound: 1.5811733
time: 4.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0232420, 3.0514815
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4206119, 3.4154634
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6771441, 2.6870251
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7584667, 3.7554116
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2551985, 3.2554297
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6623802, 2.6674881
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3472824, 3.3794026
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4430914, 3.4449782
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8605604, 2.8604486
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4218740, 2.4187081

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928328, upper bound: 1.5759583
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5920388, upper bound: 1.5767501
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0563240, 3.0183997
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4194999, 3.4165750
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6900549, 2.6741152
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7566872, 3.7571912
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2594805, 3.2511487
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6640377, 2.6658297
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3812351, 3.3454504
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4470720, 3.4409966
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8580389, 2.8629706
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4213085, 2.4192741

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5822037, upper bound: 1.5865892
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5814192, upper bound: 1.5873884
time: 4.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0360441, 3.0386796
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4208388, 3.4152360
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6764021, 2.6877680
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7519131, 3.7619653
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2574835, 3.2531452
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6648140, 2.6650543
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3582802, 3.3684053
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4460278, 3.4420414
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8533764, 2.8676324
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4214058, 2.4191768

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5866201, upper bound: 1.5821760
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5858253, upper bound: 1.5829704
time: 4.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0477591, 3.0269647
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4184985, 3.4175754
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6946049, 2.6695645
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7620201, 3.7518582
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2563429, 3.2542858
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6626606, 2.6672077
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3757629, 3.3509235
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4435053, 3.4445639
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8644724, 2.8565371
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4204721, 2.4201105

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5867415, upper bound: 1.5820497
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5859563, upper bound: 1.5828534
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0274792, 3.0472445
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4198375, 3.4162364
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6809521, 2.6832173
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7572460, 3.7566323
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2543468, 3.2562819
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6634359, 2.6664319
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3528061, 3.3738785
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4424601, 3.4456091
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8598099, 2.8611989
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4205694, 2.4200132

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5911562, upper bound: 1.5776390
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5903622, upper bound: 1.5784309
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0472441, 3.0274789
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4162364, 3.4198380
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6832170, 2.6809521
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7566319, 3.7572465
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2562819, 3.2543468
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6664314, 2.6634359
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3738785, 3.3528070
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4456091, 3.4424605
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8611984, 2.8598104
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4200134, 2.4205692

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5784319, upper bound: 1.5903622
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5776402, upper bound: 1.5911561
time: 4.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0269651, 3.0477591
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4175754, 3.4184985
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6695642, 2.6946049
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7518587, 3.7620196
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2542858, 3.2563429
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6672077, 2.6626606
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3509235, 3.3757629
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4445639, 3.4435058
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8565369, 2.8644722
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4201107, 2.4204719

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5828541, upper bound: 1.5859569
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5820504, upper bound: 1.5867415
time: 4.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0386801, 3.0360439
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4152369, 3.4208384
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6877680, 2.6764014
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7619648, 3.7519135
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2531452, 3.2574835
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6650543, 2.6648135
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3684044, 3.3582797
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4420414, 3.4460282
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8676319, 2.8533769
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4191771, 2.4214056

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5829713, upper bound: 1.5858255
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5821774, upper bound: 1.5866200
time: 5.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0183992, 3.0563240
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4165759, 3.4194994
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6741152, 2.6900542
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7571907, 3.7566867
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2511492, 3.2594800
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6658297, 2.6640382
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3454494, 3.3812361
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4409962, 3.4470730
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8629704, 2.8580387
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4192743, 2.4213083

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5873883, upper bound: 1.5814203
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5865891, upper bound: 1.5822045
time: 4.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0514812, 3.0232420
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4154639, 3.4206109
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6870251, 2.6771443
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7554121, 3.7584662
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2554293, 3.2551990
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6674881, 2.6623802
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3794022, 3.3472834
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4449778, 3.4430914
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8604479, 2.8605607
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4187078, 2.4218743

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5767512, upper bound: 1.5920388
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5759595, upper bound: 1.5928327
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0312014, 3.0435221
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4168029, 3.4192719
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6733723, 2.6907971
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7506390, 3.7632394
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2534342, 3.2571950
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6682634, 2.6616049
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3564472, 3.3702383
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4439325, 3.4441366
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8557863, 2.8652225
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4188051, 2.4217775

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5811741, upper bound: 1.5876384
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5803740, upper bound: 1.5884223
time: 4.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0429163, 3.0318069
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4144626, 3.4216113
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6915760, 2.6725936
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7607441, 3.7531333
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2522936, 3.2583356
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6661100, 2.6637578
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3739300, 3.3527565
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4414110, 3.4466591
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8668814, 2.8541269
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4178715, 2.4227107

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5812895, upper bound: 1.5875014
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5804952, upper bound: 1.5882965
time: 4.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0226364, 3.0520871
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4158015, 3.4202724
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6779232, 2.6862464
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7559719, 3.7579064
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2502966, 3.2603316
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6668854, 2.6629825
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3509731, 3.3757114
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4403658, 3.4477038
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8622198, 2.8587890
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4179688, 2.4226139

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5857061, upper bound: 1.5831008
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5849100, upper bound: 1.5838838
time: 4.58 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5838842, upper bound: 1.5849099
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5830993, upper bound: 1.5857052
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5882966, upper bound: 1.5804942
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5875011, upper bound: 1.5812886
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5884224, upper bound: 1.5803733
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5876384, upper bound: 1.5811733
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5928328, upper bound: 1.5759583
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5920388, upper bound: 1.5767501
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5822037, upper bound: 1.5865892
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5814192, upper bound: 1.5873884
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5866201, upper bound: 1.5821760
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5858253, upper bound: 1.5829704
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5867415, upper bound: 1.5820497
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5859563, upper bound: 1.5828534
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5911562, upper bound: 1.5776390
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5903622, upper bound: 1.5784309
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5784319, upper bound: 1.5903622
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5776402, upper bound: 1.5911561
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5828541, upper bound: 1.5859569
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5820504, upper bound: 1.5867415
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5829713, upper bound: 1.5858255
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5821774, upper bound: 1.5866200
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5873883, upper bound: 1.5814203
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5865891, upper bound: 1.5822045
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5767512, upper bound: 1.5920388
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5759595, upper bound: 1.5928327
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5811741, upper bound: 1.5876384
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5803740, upper bound: 1.5884223
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5812895, upper bound: 1.5875014
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5804952, upper bound: 1.5882965
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5857061, upper bound: 1.5831008
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.71
Output dim: 9, lower bound: -1.5849100, upper bound: 1.5838838

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0474434, 3.0185699
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4210939, 3.4170308
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6843500, 2.6757586
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7740231, 3.7746387
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2562070, 3.2477174
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6889467, 2.6899381
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3603992, 3.3375702
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4555836, 3.4473610
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8998199, 2.9085395
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4213712, 2.4165492

Time for backsubstitution: 14.63 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.44260835647583
rel_dist={9: [-1.592849651897651, 1.5928489343896768]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 485

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147654, upper bound: 1.4128930
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4128930, upper bound: 1.4147638
time: 6.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.38 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.38
Output dim: 9, lower bound: -1.4147654, upper bound: 1.4128930
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.38
Output dim: 9, lower bound: -1.4128930, upper bound: 1.4147638

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0445452, 3.0429313
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1871319, 3.1857872
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6991386, 2.6981285
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5890584, 3.5886340
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1389484, 3.1375990
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5274649, 2.5286145
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3899217, 3.3893113
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3251467, 3.3244486
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7522035, 2.7530067
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3391056, 2.3382399

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6238

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147650, upper bound: 1.4123091
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4141793, upper bound: 1.4128922
time: 8.65 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0429316, 3.0445454
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1857872, 3.1871324
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6981287, 2.6991382
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5886331, 3.5890594
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1375980, 3.1389489
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5286150, 2.5274644
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3893113, 3.3899221
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3244486, 3.3251472
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7530065, 2.7522035
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3382397, 2.3391063

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6238

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4128926, upper bound: 1.4141785
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4123075, upper bound: 1.4147639
time: 13.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 33.40 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 33.40
Output dim: 9, lower bound: -1.4147650, upper bound: 1.4123091
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 33.40
Output dim: 9, lower bound: -1.4141793, upper bound: 1.4128922
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 33.40
Output dim: 9, lower bound: -1.4128926, upper bound: 1.4141785
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 33.40
Output dim: 9, lower bound: -1.4123075, upper bound: 1.4147639

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0227890, 3.0225873
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1873760, 3.1857719
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6792254, 2.6794839
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5955458, 3.5947137
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1363668, 3.1347332
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5189285, 2.5204296
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3466144, 3.3478451
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3253388, 3.3244300
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7516680, 2.7522206
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3299968, 2.3286953

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4132121, upper bound: 1.4123049
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147634, upper bound: 1.4107532
time: 11.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0242023, 3.0211751
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1871176, 3.1860294
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6804938, 2.6782148
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5951395, 3.5951200
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1360826, 3.1350174
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5192795, 2.5200777
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3484550, 3.3460040
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3251281, 3.3246403
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7514172, 2.7524707
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3295619, 2.3291302

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4126263, upper bound: 1.4128903
time: 10.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4141777, upper bound: 1.4113389
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0211754, 3.0242014
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1860294, 3.1871171
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6782146, 2.6804938
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5951204, 3.5951390
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1350164, 3.1360831
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5200787, 2.5192795
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3460040, 3.3484559
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3246398, 3.3251286
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7524710, 2.7514174
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3291299, 2.3295617

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4113393, upper bound: 1.4141766
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4128910, upper bound: 1.4126285
time: 7.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0225868, 3.0227892
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1857719, 3.1873746
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6794848, 2.6792245
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5947142, 3.5955453
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1347322, 3.1363673
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5204296, 2.5189276
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3478446, 3.3466148
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3244300, 3.3253388
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7522211, 2.7516675
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3286951, 2.3299971

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4107535, upper bound: 1.4147625
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4123053, upper bound: 1.4132107
time: 6.60 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.62 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.62
Output dim: 9, lower bound: -1.4132121, upper bound: 1.4123049
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.62
Output dim: 9, lower bound: -1.4147634, upper bound: 1.4107532
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 26.62
Output dim: 9, lower bound: -1.4126263, upper bound: 1.4128903
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.62
Output dim: 9, lower bound: -1.4141777, upper bound: 1.4113389
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.62
Output dim: 9, lower bound: -1.4113393, upper bound: 1.4141766
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 26.62
Output dim: 9, lower bound: -1.4128910, upper bound: 1.4126285
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.62
Output dim: 9, lower bound: -1.4107535, upper bound: 1.4147625
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.62
Output dim: 9, lower bound: -1.4123053, upper bound: 1.4132107

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9569297, 2.9538724
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1877036, 3.1857672
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6535492, 2.6553259
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6017475, 3.6026945
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1272082, 3.1245294
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5189056, 2.5199490
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3132410, 3.3126478
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3237491, 3.3216510
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7075663, 2.7102649
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3366885, 2.3351080

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4116972, upper bound: 1.4123036
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4132112, upper bound: 1.4107898
time: 4.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9540744, 2.9567275
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1873698, 3.1861005
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6550665, 2.6538091
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6035252, 3.6009169
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1261630, 3.1255751
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5184479, 2.5204082
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3114176, 3.3144722
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3225608, 3.3228402
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7097120, 2.7081203
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3364100, 2.3353865

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4132483, upper bound: 1.4107512
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147625, upper bound: 1.4092373
time: 4.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9554868, 2.9553151
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1871123, 3.1863585
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6563358, 2.6525397
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6031189, 3.6013231
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1258788, 3.1258593
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5187988, 2.5200562
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3132582, 3.3126311
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3223510, 3.3230505
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7094612, 2.7083704
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3359742, 2.3358219

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4126626, upper bound: 1.4113374
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4141768, upper bound: 1.4098233
time: 4.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9553151, 2.9554865
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1863589, 3.1871123
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6525393, 2.6563356
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6013231, 3.6031189
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1258588, 3.1258793
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5200558, 2.5187988
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3126307, 3.3132586
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3230510, 3.3223495
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7083702, 2.7094615
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3358216, 2.3359745

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4098245, upper bound: 1.4141758
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4113384, upper bound: 1.4126620
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9567275, 2.9540744
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1861005, 3.1873698
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6538095, 2.6550663
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6009159, 3.6035261
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1255746, 3.1261635
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5204086, 2.5184469
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3144712, 3.3114176
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3228412, 3.3225598
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7081194, 2.7097116
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3353868, 2.3364098

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4092387, upper bound: 1.4147616
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4107526, upper bound: 1.4132474
time: 4.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9538732, 2.9569292
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1857677, 3.1877036
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6553259, 2.6535494
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6026936, 3.6017485
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1245284, 3.1272092
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5199490, 2.5189061
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3126478, 3.3132420
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3216510, 3.3237491
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7102652, 2.7075672
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3351083, 2.3366888

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4107902, upper bound: 1.4132103
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4123045, upper bound: 1.4116961
time: 4.51 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.09 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.09
Output dim: 9, lower bound: -1.4116972, upper bound: 1.4123036
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.09
Output dim: 9, lower bound: -1.4132112, upper bound: 1.4107898
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.09
Output dim: 9, lower bound: -1.4132483, upper bound: 1.4107512
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.09
Output dim: 9, lower bound: -1.4147625, upper bound: 1.4092373
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.09
Output dim: 9, lower bound: -1.4126626, upper bound: 1.4113374
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.09
Output dim: 9, lower bound: -1.4141768, upper bound: 1.4098233
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.09
Output dim: 9, lower bound: -1.4098245, upper bound: 1.4141758
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.09
Output dim: 9, lower bound: -1.4113384, upper bound: 1.4126620
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.09
Output dim: 9, lower bound: -1.4092387, upper bound: 1.4147616
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.09
Output dim: 9, lower bound: -1.4107526, upper bound: 1.4132474
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.09
Output dim: 9, lower bound: -1.4107902, upper bound: 1.4132103
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.09
Output dim: 9, lower bound: -1.4123045, upper bound: 1.4116961

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.7946835, 2.7983871
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1957622, 3.1933794
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5442600, 2.5505874
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5635195, 3.5660567
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1424103, 3.1403961
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5129452, 2.5137296
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1295986, 3.1366563
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3316870, 3.3299379
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6702013, 2.6744533
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3359451, 2.3343322

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4132107, upper bound: 1.4104516
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4128731, upper bound: 1.4107892
time: 4.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.7985888, 2.7944820
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1949821, 3.1941590
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5503283, 2.5445197
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5668888, 3.5626874
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1420298, 3.1407762
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5122280, 2.5144472
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1354256, 3.1308289
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3308458, 3.3307786
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6738997, 2.6707549
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3356342, 2.3346431

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4132478, upper bound: 1.4104123
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4129109, upper bound: 1.4107511
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.7918291, 2.8012419
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1954284, 3.1937127
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5457773, 2.5490706
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5652971, 3.5642791
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1413641, 3.1414413
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5124855, 2.5141888
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1277733, 3.1384807
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3304977, 3.3311272
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6723461, 2.6723089
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3356667, 2.3346107

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147621, upper bound: 1.4088989
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4144251, upper bound: 1.4092372
time: 4.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.7932405, 2.7998297
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1951709, 3.1939707
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5470467, 2.5478013
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5648899, 3.5646863
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1410799, 3.1417255
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5128384, 2.5138369
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1296158, 3.1366391
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3302870, 3.3313375
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6720953, 2.6725590
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3352308, 2.3350461

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4141764, upper bound: 1.4094848
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4138395, upper bound: 1.4098228
time: 4.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.7998295, 2.7932410
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1939712, 3.1951709
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5478010, 2.5470462
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5646858, 3.5648904
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1417255, 3.1410804
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5138369, 2.5128379
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1366386, 3.1296153
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3313370, 3.3302879
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6725588, 2.6720960
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3350458, 2.3352311

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4098240, upper bound: 1.4138388
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4094860, upper bound: 1.4141780
time: 5.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.8012419, 2.7918289
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1937137, 3.1954288
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5490713, 2.5457768
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5642796, 3.5652966
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1414413, 3.1413646
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5141888, 2.5124860
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1384811, 3.1277742
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3311262, 3.3304982
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6723089, 2.6723461
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3346109, 2.3356664

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4092382, upper bound: 1.4144245
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4089000, upper bound: 1.4147638
time: 4.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.7944822, 2.7985888
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1941600, 3.1949821
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5445194, 2.5503280
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5626879, 3.5668883
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1407766, 3.1420298
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5144472, 2.5122275
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1308289, 3.1354260
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3307781, 3.3308463
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6707544, 2.6739001
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3346434, 2.3356340

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4107521, upper bound: 1.4129102
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4104135, upper bound: 1.4132495
time: 4.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.7983866, 2.7946837
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1933799, 3.1957622
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5505877, 2.5442600
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5660572, 3.5635190
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1403961, 3.1424103
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5137291, 2.5129452
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1366558, 3.1295986
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3299379, 3.3316875
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6744528, 2.6702018
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3343325, 2.3359449

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4107897, upper bound: 1.4128719
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4104524, upper bound: 1.4132094
time: 8.06 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 27.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.65
Output dim: 9, lower bound: -1.4132107, upper bound: 1.4104516
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.65
Output dim: 9, lower bound: -1.4128731, upper bound: 1.4107892
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.65
Output dim: 9, lower bound: -1.4132478, upper bound: 1.4104123
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.65
Output dim: 9, lower bound: -1.4129109, upper bound: 1.4107511
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.65
Output dim: 9, lower bound: -1.4147621, upper bound: 1.4088989
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.65
Output dim: 9, lower bound: -1.4144251, upper bound: 1.4092372
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.65
Output dim: 9, lower bound: -1.4141764, upper bound: 1.4094848
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.65
Output dim: 9, lower bound: -1.4138395, upper bound: 1.4098228
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.65
Output dim: 9, lower bound: -1.4098240, upper bound: 1.4138388
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.65
Output dim: 9, lower bound: -1.4094860, upper bound: 1.4141780
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.65
Output dim: 9, lower bound: -1.4092382, upper bound: 1.4144245
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.65
Output dim: 9, lower bound: -1.4089000, upper bound: 1.4147638
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 27.65
Output dim: 9, lower bound: -1.4107521, upper bound: 1.4129102
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.65
Output dim: 9, lower bound: -1.4104135, upper bound: 1.4132495
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 27.65
Output dim: 9, lower bound: -1.4107897, upper bound: 1.4128719
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.65
Output dim: 9, lower bound: -1.4104524, upper bound: 1.4132094

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.7900400, 2.7939355
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1965837, 3.1943369
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5421848, 2.5484231
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5796356, 3.5830226
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1382852, 3.1367865
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5369682, 2.5367823
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1142864, 3.1219807
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3389778, 3.3369331
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7112327, 2.7172472
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3345850, 2.3329129

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4131079, upper bound: 1.4104514
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4132105, upper bound: 1.4103495
time: 4.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.7939453, 2.7900305
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1958036, 3.1951170
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5482531, 2.5423553
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5830040, 3.5796542
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1379046, 3.1371665
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5362501, 2.5374999
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1201143, 3.1161528
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3381367, 3.3377738
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7149310, 2.7135489
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3342731, 2.3332238

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4131450, upper bound: 1.4104121
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4132481, upper bound: 1.4103101
time: 4.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.7871847, 2.7967904
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1962509, 3.1946707
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5437021, 2.5469062
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5814133, 3.5812449
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1372399, 3.1378322
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5365095, 2.5372415
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1124630, 3.1238046
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3377876, 3.3381224
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7133765, 2.7151029
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3343055, 2.3331914

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4146593, upper bound: 1.4088988
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147618, upper bound: 1.4087965
time: 4.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.7873783, 2.7965980
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1963873, 3.1945348
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5436134, 2.5469952
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5822630, 3.5803943
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1377549, 3.1373167
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5355387, 2.5382118
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1130981, 3.1231685
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3374939, 3.3384171
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7151389, 2.7133403
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3342474, 2.3332505

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4143223, upper bound: 1.4092369
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4144249, upper bound: 1.4091344
time: 4.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.7885971, 2.7953782
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1959925, 3.1949282
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5449715, 2.5456369
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5810061, 3.5816512
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1369557, 3.1381159
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5368605, 2.5368896
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1143036, 3.1219635
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3375778, 3.3383327
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7131267, 2.7153530
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3338706, 2.3336267

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4140736, upper bound: 1.4094847
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4141762, upper bound: 1.4093825
time: 4.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.7887907, 2.7951858
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1961288, 3.1947927
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5448818, 2.5457261
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5818558, 3.5808015
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1374707, 3.1376009
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5358906, 2.5378599
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1149397, 3.1213269
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3372831, 3.3386273
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7148890, 2.7135904
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3338115, 2.3336854

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4137366, upper bound: 1.4098228
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4138392, upper bound: 1.4097199
time: 4.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.7951860, 2.7887900
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1947927, 3.1961288
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5457268, 2.5448818
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5808010, 3.5818563
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1376004, 3.1374707
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5378599, 2.5358906
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1213274, 3.1149392
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3386269, 3.3372831
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7135901, 2.7148898
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3336856, 2.3338118

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4097211, upper bound: 1.4138380
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4098237, upper bound: 1.4137361
time: 4.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.7953787, 2.7885971
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1949282, 3.1959929
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5456371, 2.5449710
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5816517, 3.5810065
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1381154, 3.1369557
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5368891, 2.5368609
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1219635, 3.1143031
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3383331, 3.3375778
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7153525, 2.7131274
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3336265, 2.3338709

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4093834, upper bound: 1.4141755
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4094857, upper bound: 1.4140752
time: 4.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.7965984, 2.7873778
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1945353, 3.1963868
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5469952, 2.5436125
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5803947, 3.5822635
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1373162, 3.1377549
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5382118, 2.5355387
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1231689, 3.1130981
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3384171, 3.3374934
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7133403, 2.7151399
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3332508, 2.3342471

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4091354, upper bound: 1.4144239
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4092379, upper bound: 1.4143215
time: 4.74 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 24.10 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4131079, upper bound: 1.4104514
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4132105, upper bound: 1.4103495
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4131450, upper bound: 1.4104121
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4132481, upper bound: 1.4103101
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4146593, upper bound: 1.4088988
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4147618, upper bound: 1.4087965
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4143223, upper bound: 1.4092369
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4144249, upper bound: 1.4091344
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4140736, upper bound: 1.4094847
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4141762, upper bound: 1.4093825
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4137366, upper bound: 1.4098228
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4138392, upper bound: 1.4097199
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4097211, upper bound: 1.4138380
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4098237, upper bound: 1.4137361
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4093834, upper bound: 1.4141755
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4094857, upper bound: 1.4140752
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4091354, upper bound: 1.4144239
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.10
Output dim: 9, lower bound: -1.4092379, upper bound: 1.4143215
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.4089000, upper bound: 1.4147638
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.4104135, upper bound: 1.4132495
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.3590686321258545
rel_dist={9: [-1.414766908177059, 1.4147664541724545]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 2405.95 seconds
