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
execution time: IAR + LP analysis = 15.20 + 33.34 = 48.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -2.1425934, upper bound: 2.1425903


# Binary Search by BASE starts (time budget: 3551.46 seconds, max iter: 100)

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
Binary search time: 151.16 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 3400.30 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6126

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240692, upper bound: 1.8222131
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8222130, upper bound: 1.8240691
time: 5.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.00 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.00
Output dim: 9, lower bound: -1.8240692, upper bound: 1.8222131
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.00
Output dim: 9, lower bound: -1.8222130, upper bound: 1.8240691

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6558762, 3.6487157
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7714076, 3.7702262
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0382147, 3.0381415
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0482502, 4.0486140
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4378295, 3.4559908
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9248123, 2.9326522
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.9193416, 3.9218831
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5898056, 3.5861220
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1912584, 3.1921105
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5718284, 2.5644157

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8230523, upper bound: 1.8222042
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240603, upper bound: 1.8211881
time: 4.58 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6487160, 3.6520987
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7702260, 3.7707870
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0381413, 3.0381792
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0484219, 4.0482497
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4464107, 3.4378309
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9285202, 2.9248126
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.9205413, 3.9193420
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5861225, 3.5878572
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1916580, 3.1912584
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5644155, 2.5679176

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8211883, upper bound: 1.8240601
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8222041, upper bound: 1.8230521
time: 4.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.43 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.43
Output dim: 9, lower bound: -1.8230523, upper bound: 1.8222042
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.43
Output dim: 9, lower bound: -1.8240603, upper bound: 1.8211881
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.43
Output dim: 9, lower bound: -1.8211883, upper bound: 1.8240601
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.43
Output dim: 9, lower bound: -1.8222041, upper bound: 1.8230521

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6607900, 3.6507077
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7606411, 3.7621484
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0434270, 3.0446634
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0492964, 4.0499144
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4310818, 3.4519353
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9392924, 2.9442081
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8767023, 3.8899083
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5874934, 3.5807457
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1803341, 3.1911058
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5803790, 2.5762396

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8152337, upper bound: 1.8221970
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8230490, upper bound: 1.8143252
time: 4.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6578679, 3.6536298
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7633305, 3.7594595
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0447373, 3.0433531
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0495501, 4.0496597
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4337749, 3.4492412
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9363685, 2.9471321
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8873672, 3.8792439
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5844302, 3.5838094
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1902533, 3.1811867
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5836530, 2.5729656

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8235975, upper bound: 1.8199822
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8228691, upper bound: 1.8207384
time: 4.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6536298, 3.6540911
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7594604, 3.7627096
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0433526, 3.0447001
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0494661, 4.0495501
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4396610, 3.4337754
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9430003, 2.9363685
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8779001, 3.8873677
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5838094, 3.5824790
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1807356, 3.1902535
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5729651, 2.5797420

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 485

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8133418, upper bound: 1.8240531
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8211850, upper bound: 1.8162128
time: 5.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6507077, 3.6570129
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7621479, 3.7600207
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0446639, 3.0433898
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0497208, 4.0492954
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4423561, 3.4310813
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9400764, 2.9392924
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8885651, 3.8767028
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5807452, 3.5855427
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1906548, 3.1803346
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5762391, 2.5764680

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 485

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8221929, upper bound: 1.8121758
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8112592, upper bound: 1.8230408
time: 4.64 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.06 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.06
Output dim: 9, lower bound: -1.8152337, upper bound: 1.8221970
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.06
Output dim: 9, lower bound: -1.8230490, upper bound: 1.8143252
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.06
Output dim: 9, lower bound: -1.8235975, upper bound: 1.8199822
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.06
Output dim: 9, lower bound: -1.8228691, upper bound: 1.8207384
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.06
Output dim: 9, lower bound: -1.8133418, upper bound: 1.8240531
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.06
Output dim: 9, lower bound: -1.8211850, upper bound: 1.8162128
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.06
Output dim: 9, lower bound: -1.8221929, upper bound: 1.8121758
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.06
Output dim: 9, lower bound: -1.8112592, upper bound: 1.8230408

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6092043, 3.5819931
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7626367, 3.7621429
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0177517, 3.0280900
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0554986, 4.0667820
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4271517, 3.4417324
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9415679, 2.9437284
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8524528, 3.8547120
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5918489, 3.5779667
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1362319, 3.1598713
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5884650, 2.5826528

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6238

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8152302, upper bound: 1.8193575
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8124137, upper bound: 1.8221931
time: 4.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5920744, 3.5991230
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7606359, 3.7641437
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0268536, 3.0189886
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0661635, 4.0561171
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4208784, 3.4480057
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9388118, 2.9464836
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8415065, 3.8656583
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5847135, 3.5851016
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1490998, 3.1470041
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5867922, 2.5843256

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8141955, upper bound: 1.8143194
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8230428, upper bound: 1.8055078
time: 4.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6571279, 3.6524634
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7596779, 3.7545712
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0473604, 3.0453835
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0536089, 4.0528021
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4356518, 3.4516497
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9356256, 2.9457517
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8880672, 3.8797932
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5844393, 3.5838208
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1938820, 3.1840053
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5861125, 2.5761404

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8157509, upper bound: 1.8199794
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8235898, upper bound: 1.8121433
time: 4.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6567016, 3.6528895
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7584419, 3.7558067
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0467682, 3.0459757
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0526934, 4.0537176
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4361839, 3.4511175
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9349875, 2.9463897
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8879175, 3.8799434
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5844412, 3.5838180
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1930714, 3.1848159
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5868278, 2.5754251

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8228672, upper bound: 1.8196568
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8217243, upper bound: 1.8207363
time: 4.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6020451, 3.5853763
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7614551, 3.7627037
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0176783, 3.0281267
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0556693, 4.0664177
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4357328, 3.4235725
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9452758, 2.9358888
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8536515, 3.8521714
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5881658, 3.5797009
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1366334, 3.1590190
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5810521, 2.5861552

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8133284, upper bound: 1.8216957
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8109959, upper bound: 1.8240398
time: 7.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5849152, 3.6025062
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7594543, 3.7647045
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0267801, 3.0190253
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0663352, 4.0557528
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4294586, 3.4298463
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9425197, 2.9386444
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8427052, 3.8631172
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5810304, 3.5868359
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1495004, 3.1461518
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5793793, 2.5878279

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8211833, upper bound: 1.8150157
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8200957, upper bound: 1.8162111
time: 4.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6216888, 3.6183085
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7379303, 3.7277298
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0264583, 3.0191264
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0623617, 4.0593877
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4180593, 3.3986864
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9124622, 2.9185781
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.9067202, 3.8911905
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5984674, 3.5990753
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.2097611, 3.2042615
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5606110, 2.5556390

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8221797, upper bound: 1.8099860
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8198336, upper bound: 1.8121624
time: 4.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6120043, 3.6279907
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7298584, 3.7358017
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0203996, 3.0251851
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0598116, 4.0619369
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4099588, 3.4067864
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9193630, 2.9116788
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.9030542, 3.8948569
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5942769, 3.6032648
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.2145810, 3.1994414
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5554106, 2.5608404

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6139

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8108499, upper bound: 1.8230401
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8112586, upper bound: 1.8227110
time: 4.88 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 9, lower bound: -1.8152302, upper bound: 1.8193575
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 9, lower bound: -1.8124137, upper bound: 1.8221931
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 9, lower bound: -1.8141955, upper bound: 1.8143194
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 9, lower bound: -1.8230428, upper bound: 1.8055078
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 9, lower bound: -1.8157509, upper bound: 1.8199794
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 9, lower bound: -1.8235898, upper bound: 1.8121433
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 9, lower bound: -1.8228672, upper bound: 1.8196568
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 9, lower bound: -1.8217243, upper bound: 1.8207363
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 9, lower bound: -1.8133284, upper bound: 1.8216957
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 9, lower bound: -1.8109959, upper bound: 1.8240398
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 9, lower bound: -1.8211833, upper bound: 1.8150157
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 9, lower bound: -1.8200957, upper bound: 1.8162111
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 9, lower bound: -1.8221797, upper bound: 1.8099860
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 9, lower bound: -1.8198336, upper bound: 1.8121624
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 9, lower bound: -1.8108499, upper bound: 1.8230401
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.00
Output dim: 9, lower bound: -1.8112586, upper bound: 1.8227110

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5874481, 3.5687103
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7641668, 3.7621276
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9978380, 3.0157917
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0640192, 4.0728621
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4259901, 3.4388661
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9330292, 2.9373016
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8091459, 3.8224516
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5930920, 3.5779481
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1369467, 3.1590853
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5815310, 2.5731089

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 485

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8152191, upper bound: 1.8083571
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8043256, upper bound: 1.8193462
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5959225, 3.5602365
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7626209, 3.7636735
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0054541, 3.0081761
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0615788, 4.0753026
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4242849, 3.4405699
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9351406, 2.9351902
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8201914, 3.8114047
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5918312, 3.5792098
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1354465, 3.1605856
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5789208, 2.5757191

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6139

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8120264, upper bound: 1.8221927
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8124132, upper bound: 1.8218117
time: 7.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.4703884, 3.4368763
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7682476, 3.7744341
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9448667, 2.9096959
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0374823, 4.0178890
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4400725, 3.4632072
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9325924, 2.9418147
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.7037735, 3.6820149
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5947428, 3.5930409
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1210594, 3.1096399
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5860162, 2.5837436

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6139

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8138805, upper bound: 1.8143183
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8141950, upper bound: 1.8139577
time: 4.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.4298277, 3.4774361
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7709255, 3.7717562
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9175601, 2.9370015
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0279360, 4.0274353
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4360805, 3.4671998
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9341450, 2.9402640
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6578636, 3.7279243
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5926533, 3.5951309
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1117353, 3.1189632
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5862103, 2.5835495

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8225927, upper bound: 1.8043093
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8218369, upper bound: 1.8050389
time: 6.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6055422, 3.5837483
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7616730, 3.7545662
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0216851, 3.0288103
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0598092, 4.0696692
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4317245, 3.4414487
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9379029, 2.9452722
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8638172, 3.8445969
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5887942, 3.5810423
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1497812, 3.1527710
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5941985, 2.5825534

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8141087, upper bound: 1.8192282
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8145702, upper bound: 1.8192412
time: 4.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5884123, 3.6008782
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7596722, 3.7565670
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0307870, 3.0197089
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0704751, 4.0590034
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4254494, 3.4477220
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9351468, 2.9480274
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8528709, 3.8555427
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5816598, 3.5881767
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1626482, 3.1399035
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5925257, 2.5842261

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8147788, upper bound: 1.8121372
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8235837, upper bound: 1.8032856
time: 4.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6520567, 3.6493981
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7592649, 3.7574437
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0451384, 3.0438116
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0688052, 4.0749335
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4320636, 3.4500875
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9638634, 2.9694431
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8725991, 3.8684440
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5932088, 3.5908151
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.2341223, 3.2364402
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5857615, 2.5740061

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 485

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8228561, upper bound: 1.8086849
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8118870, upper bound: 1.8196459
time: 7.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6532116, 3.6482451
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7600794, 3.7566297
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0446043, 3.0443466
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0739093, 4.0698314
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4351554, 3.4469972
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9580412, 2.9752676
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8764234, 3.8646264
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5914388, 3.5925860
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.2447004, 3.2258658
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5854082, 2.5743599

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8217115, upper bound: 1.8183906
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8194983, upper bound: 1.8207228
time: 4.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6055613, 3.5838032
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7601666, 3.7655902
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0169802, 3.0296850
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0564280, 4.0660896
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4326773, 3.4303899
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9463153, 2.9354377
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8539729, 3.8520284
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5920534, 3.5779953
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1359286, 3.1606669
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5833211, 2.5851333

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6139

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8130025, upper bound: 1.8216951
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8133279, upper bound: 1.8213126
time: 4.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6004734, 3.5853763
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7614551, 3.7614150
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0176783, 3.0274286
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0553408, 4.0664177
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4357328, 3.4205184
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9448247, 2.9358888
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8535075, 3.8521714
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5864601, 3.5797009
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1366334, 3.1583135
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5800300, 2.5861552

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8109943, upper bound: 1.8229054
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8099665, upper bound: 1.8240383
time: 5.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5802689, 3.5990157
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7602763, 3.7663422
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0251493, 3.0168600
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0824504, 4.0769720
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4253368, 3.4288173
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9713979, 2.9616976
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8273897, 3.8516235
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5897975, 3.5938320
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1905513, 3.1977818
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5783131, 2.5864086

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 568

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8211800, upper bound: 1.8140871
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8202468, upper bound: 1.8150120
time: 4.37 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5814228, 3.5978613
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7610908, 3.7655272
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0246143, 3.0173945
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0875525, 4.0718679
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4284286, 3.4257255
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9655747, 2.9675198
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8312063, 3.8477998
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5880265, 3.5956020
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.2011256, 3.1872032
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5779593, 2.5867615

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 568

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8200923, upper bound: 1.8152845
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8191605, upper bound: 1.8162077
time: 4.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6252022, 3.6167364
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7366419, 3.7306156
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0257592, 3.0206842
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0631180, 4.0590568
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4150057, 3.4055018
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9135008, 2.9181271
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.9070425, 3.8910475
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6023583, 3.5973701
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.2090573, 3.2059095
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5628810, 2.5546176

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8133444, upper bound: 1.8099799
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8221736, upper bound: 1.8012167
time: 4.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6201153, 3.6183085
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7379303, 3.7264414
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0264583, 3.0184278
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0620308, 4.0593877
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4180593, 3.3956313
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9120111, 2.9185781
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.9065781, 3.8911905
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.5967631, 3.5990753
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.2097611, 3.2035565
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5595899, 2.5556390

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8183227, upper bound: 1.8109311
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8079104, upper bound: 1.8109311
time: 4.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5972958, 3.6169605
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7231889, 3.7269213
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0008879, 3.0165164
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0688534, 4.0766258
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4055634, 3.4047866
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9140420, 2.9045832
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.9001598, 3.9006886
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6015439, 3.6090217
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.2132082, 3.1968670
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5657210, 2.5685108

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8108363, upper bound: 1.8206962
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8086406, upper bound: 1.8230267
time: 4.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.6009731, 3.6132829
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7209764, 3.7291324
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.0117311, 3.0056727
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0745020, 4.0709763
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4079609, 3.4023890
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9122682, 2.9063575
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.9088860, 3.8919616
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6000342, 3.6105313
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.2120066, 3.1980684
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5630813, 2.5711501

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 568

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8112552, upper bound: 1.8217746
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8103162, upper bound: 1.8227079
time: 4.76 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8152191, upper bound: 1.8083571
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8043256, upper bound: 1.8193462
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8120264, upper bound: 1.8221927
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8124132, upper bound: 1.8218117
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8138805, upper bound: 1.8143183
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8141950, upper bound: 1.8139577
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8225927, upper bound: 1.8043093
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8218369, upper bound: 1.8050389
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8141087, upper bound: 1.8192282
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8145702, upper bound: 1.8192412
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8147788, upper bound: 1.8121372
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8235837, upper bound: 1.8032856
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8228561, upper bound: 1.8086849
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8118870, upper bound: 1.8196459
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8217115, upper bound: 1.8183906
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8194983, upper bound: 1.8207228
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8130025, upper bound: 1.8216951
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8133279, upper bound: 1.8213126
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8109943, upper bound: 1.8229054
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8099665, upper bound: 1.8240383
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8211800, upper bound: 1.8140871
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8202468, upper bound: 1.8150120
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8200923, upper bound: 1.8152845
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8191605, upper bound: 1.8162077
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8133444, upper bound: 1.8099799
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8221736, upper bound: 1.8012167
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8183227, upper bound: 1.8109311
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8079104, upper bound: 1.8109311
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8108363, upper bound: 1.8206962
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8086406, upper bound: 1.8230267
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8112552, upper bound: 1.8217746
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.21
Output dim: 9, lower bound: -1.8103162, upper bound: 1.8227079

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.5584259, 3.5300059
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.7399483, 3.7298369
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.9796319, 2.9915276
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.0766573, 4.0829515
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.4016938, 3.4064703
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.9054151, 2.9165869
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.8272996, 3.8369403
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6108127, 3.5914803
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.1560545, 3.1830127
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.5659018, 2.5522790

Time for backsubstitution: 14.64 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.567917585372925
rel_dist={9: [-1.824091324888559, 1.824091281312575]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5924604, upper bound: 1.5920333
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5920340, upper bound: 1.5924600
time: 4.90 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.54 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.54
Output dim: 9, lower bound: -1.5924604, upper bound: 1.5920333
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.54
Output dim: 9, lower bound: -1.5920340, upper bound: 1.5924600

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.3088670, 3.3086545
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4348917, 3.4342737
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8510346, 2.8507383
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7700958, 3.7696390
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2824383, 3.2827048
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.7033958, 2.7030759
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5937347, 3.5936589
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4217029, 3.4217038
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9197445, 2.9193392
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4450688, 2.4454260

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5924440, upper bound: 1.5895083
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5899335, upper bound: 1.5920188
time: 4.96 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.3086553, 3.3088675
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4342737, 3.4348917
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8507390, 2.8510342
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7696381, 3.7700968
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2827034, 3.2824388
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.7030754, 2.7033949
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5936584, 3.5937343
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4217038, 3.4217024
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9193392, 2.9197445
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4454265, 2.4450684

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 485

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5920286, upper bound: 1.5870120
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5865857, upper bound: 1.5924549
time: 4.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.23 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.23
Output dim: 9, lower bound: -1.5924440, upper bound: 1.5895083
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.23
Output dim: 9, lower bound: -1.5899335, upper bound: 1.5920188
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.23
Output dim: 9, lower bound: -1.5920286, upper bound: 1.5870120
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.23
Output dim: 9, lower bound: -1.5865857, upper bound: 1.5924549

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.3098392, 3.3070824
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4336033, 3.4350731
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8503366, 2.8511686
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7703104, 3.7693090
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2793827, 3.2845850
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.7036886, 2.7026248
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5938234, 3.5935159
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4227943, 3.4199986
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9190397, 2.9198108
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4456925, 2.4444048

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5880317, upper bound: 1.5895052
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5924410, upper bound: 1.5850941
time: 5.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.3072948, 3.3086545
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4348917, 3.4329855
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8510346, 2.8500407
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7697659, 3.7696390
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2824383, 3.2796497
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.7029428, 2.7030759
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5935907, 3.5936589
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4199972, 3.4217038
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9197445, 2.9186344
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4440475, 2.4454260

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 485

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5899144, upper bound: 1.5903093
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5882246, upper bound: 1.5919997
time: 4.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2747931, 3.2701638
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4060192, 3.4026012
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8295031, 2.8267696
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7810025, 3.7801852
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2543588, 3.2500424
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6754627, 2.6792307
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6099815, 3.6082230
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4373298, 3.4352331
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9384489, 2.9412637
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4271972, 2.4242392

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5909473, upper bound: 1.5859535
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5856934, upper bound: 1.5859552
time: 4.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2699504, 3.2750063
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4019833, 3.4066372
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8264742, 2.8297987
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7797275, 3.7814603
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2503085, 3.2540922
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6789112, 2.6757808
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6081486, 3.6100559
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4352345, 3.4373279
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9408579, 2.9388537
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4245965, 2.4268394

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 568

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5865152, upper bound: 1.5919531
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5865109, upper bound: 1.5924539
time: 4.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.92 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.92
Output dim: 9, lower bound: -1.5880317, upper bound: 1.5895052
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.92
Output dim: 9, lower bound: -1.5924410, upper bound: 1.5850941
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.92
Output dim: 9, lower bound: -1.5899144, upper bound: 1.5903093
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.92
Output dim: 9, lower bound: -1.5882246, upper bound: 1.5919997
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.92
Output dim: 9, lower bound: -1.5909473, upper bound: 1.5859535
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.92
Output dim: 9, lower bound: -1.5856934, upper bound: 1.5859552
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.92
Output dim: 9, lower bound: -1.5865152, upper bound: 1.5919531
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.92
Output dim: 9, lower bound: -1.5865109, upper bound: 1.5924539

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1678734, 3.1448364
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4412150, 3.4440241
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7546968, 2.7418764
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7368565, 3.7310820
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2965813, 3.2997861
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6974688, 2.6971805
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.4331341, 3.4098716
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4317775, 3.4279366
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8863363, 2.8824463
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4449162, 2.4437251

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5869647, upper bound: 1.5835459
time: 5.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5820627, upper bound: 1.5884308
time: 4.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1475925, 3.1651163
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4425540, 3.4426851
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7410440, 2.7555292
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7320833, 3.7358551
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2945843, 3.3017821
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6982441, 2.6964052
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.4101791, 3.4328265
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4307323, 3.4289813
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8816748, 2.8871081
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4450130, 2.4436283

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 568

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6238

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5924393, upper bound: 1.5834135
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5907633, upper bound: 1.5850918
time: 4.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.3074923, 3.3052719
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4349213, 3.4324243
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8510346, 2.8500035
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7695932, 3.7696486
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2738571, 3.2801495
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6992350, 2.7032866
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5923920, 3.5937304
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4201050, 3.4199696
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9193439, 2.9186590
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4442515, 2.4419239

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 485

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5899090, upper bound: 1.5848651
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5844641, upper bound: 1.5903026
time: 4.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.3039122, 3.3086545
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4343300, 3.4329855
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8509974, 2.8500407
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7697659, 3.7694654
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2824383, 3.2710695
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.7029428, 2.6993670
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5935907, 3.5924602
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4182625, 3.4217038
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9197445, 2.9182329
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4405446, 2.4454260

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838215, upper bound: 1.5919967
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5882216, upper bound: 1.5875881
time: 4.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2747893, 3.2701600
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4060173, 3.4025984
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8295031, 2.8267703
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7809925, 3.7801762
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2543521, 3.2500358
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6754670, 2.6792350
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6099920, 3.6082315
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4373293, 3.4352326
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9384537, 2.9412704
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4271922, 2.4242344

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5909460, upper bound: 1.5851633
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5901750, upper bound: 1.5859523
time: 4.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2747893, 3.2701602
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4060173, 3.4025989
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8295031, 2.8267701
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7809906, 3.7801743
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2543521, 3.2500348
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6754670, 2.6792340
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6099901, 3.6082330
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4373293, 3.4352326
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9384527, 2.9412694
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4271922, 2.4242339

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5811515, upper bound: 1.5859495
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5856878, upper bound: 1.5814123
time: 4.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2768593, 3.2824259
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.3856220, 3.3879380
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8103080, 2.8152437
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7705517, 3.7725000
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2411151, 3.2442060
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6868768, 2.6827431
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5731544, 3.5792832
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4328303, 3.4345841
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9346013, 2.9325650
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4204345, 2.4229007

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5857600, upper bound: 1.5914408
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5859931, upper bound: 1.5912078
time: 4.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2777977, 3.2819145
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.3832836, 3.3902774
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8125091, 2.8136332
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7716980, 3.7722836
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2404218, 3.2454424
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6858735, 2.6837463
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5775318, 3.5750623
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4324908, 3.4349265
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9353843, 2.9325972
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4206581, 2.4233918

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5820910, upper bound: 1.5924509
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5865079, upper bound: 1.5880427
time: 4.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.96
Output dim: 9, lower bound: -1.5869647, upper bound: 1.5835459
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.96
Output dim: 9, lower bound: -1.5820627, upper bound: 1.5884308
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.96
Output dim: 9, lower bound: -1.5924393, upper bound: 1.5834135
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.96
Output dim: 9, lower bound: -1.5907633, upper bound: 1.5850918
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.96
Output dim: 9, lower bound: -1.5899090, upper bound: 1.5848651
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.96
Output dim: 9, lower bound: -1.5844641, upper bound: 1.5903026
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.96
Output dim: 9, lower bound: -1.5838215, upper bound: 1.5919967
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.96
Output dim: 9, lower bound: -1.5882216, upper bound: 1.5875881
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.96
Output dim: 9, lower bound: -1.5909460, upper bound: 1.5851633
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.96
Output dim: 9, lower bound: -1.5901750, upper bound: 1.5859523
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.96
Output dim: 9, lower bound: -1.5811515, upper bound: 1.5859495
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.96
Output dim: 9, lower bound: -1.5856878, upper bound: 1.5814123
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.96
Output dim: 9, lower bound: -1.5857600, upper bound: 1.5914408
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.96
Output dim: 9, lower bound: -1.5859931, upper bound: 1.5912078
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.96
Output dim: 9, lower bound: -1.5820910, upper bound: 1.5924509
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.96
Output dim: 9, lower bound: -1.5865079, upper bound: 1.5880427

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1678705, 3.1448338
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4412131, 3.4440215
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7546973, 2.7418766
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7368445, 3.7310715
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2965727, 3.2997789
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6974745, 2.6971846
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.4331455, 3.4098811
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4317775, 3.4279361
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8863420, 2.8824506
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4449124, 2.4437218

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5869634, upper bound: 1.5827714
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5861853, upper bound: 1.5835421
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1678705, 3.1448338
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4412131, 3.4440219
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7546973, 2.7418766
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7368464, 3.7310696
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2965736, 3.2997780
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6974726, 2.6971850
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.4331436, 3.4098825
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4317775, 3.4279361
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8863411, 2.8824515
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4449129, 2.4437213

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6139

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 485

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5816786, upper bound: 1.5829957
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5815138, upper bound: 1.5884236
time: 4.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1258254, 3.1475863
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4433155, 3.4426703
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7211189, 2.7394111
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7393899, 3.7419424
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2925630, 3.2989087
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6897039, 2.6889210
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3668690, 3.3950415
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4313459, 3.4289641
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8816471, 2.8863294
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4367652, 2.4340758

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 485

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5924339, upper bound: 1.5779565
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5869920, upper bound: 1.5834073
time: 4.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1300635, 3.1433492
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4425392, 3.4434433
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7249260, 2.7356033
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7381711, 3.7431622
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2917113, 3.2997594
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6907587, 2.6878653
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3723927, 3.3895168
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4307146, 3.4295950
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8808956, 2.8870797
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4354601, 2.4353790

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 471

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5907443, upper bound: 1.5833945
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5890540, upper bound: 1.5850730
time: 5.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2736311, 3.2665679
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4066653, 3.4001327
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8297992, 2.8257389
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7809601, 3.7797379
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2455115, 3.2477536
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6716213, 2.6791224
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6087136, 3.6082187
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4357300, 3.4335008
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9384518, 2.9401779
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4260225, 2.4210947

Time for backsubstitution: 14.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6139

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5895516, upper bound: 1.5848644
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5899084, upper bound: 1.5845076
time: 4.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2687893, 3.2714102
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4026294, 3.4041686
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8267694, 2.8287680
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7796860, 3.7810121
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2414613, 3.2518034
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6750708, 2.6756730
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.6068807, 3.6100521
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4336357, 3.4355955
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9408627, 2.9377677
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4234228, 2.4236948

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5844628, upper bound: 1.5895083
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5836708, upper bound: 1.5903014
time: 5.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1619473, 3.1464086
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4419436, 3.4419370
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7553587, 2.7407484
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7363129, 3.7312393
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2996359, 3.2862711
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6967239, 2.6939237
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.4329023, 3.4088159
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4272470, 3.4296422
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8870430, 2.8808692
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4397683, 2.4447465

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5833092, upper bound: 1.5919958
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838205, upper bound: 1.5914880
time: 4.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1416683, 3.1666884
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4432826, 3.4405975
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7417059, 2.7544012
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7315397, 3.7360125
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2976398, 3.2882671
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6974993, 2.6931484
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.4099474, 3.4317708
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4262018, 3.4306870
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8823795, 2.8855309
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4398651, 2.4446497

Time for backsubstitution: 14.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 568

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5871502, upper bound: 1.5816169
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5822671, upper bound: 1.5865180
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2701478, 3.2660954
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4068398, 3.4038284
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8276062, 2.8246057
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7971067, 3.7988424
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2502289, 3.2474570
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.7014298, 2.7022877
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5946794, 3.5948277
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4452090, 3.4422278
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9794846, 2.9875891
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4259517, 2.4228165

Time for backsubstitution: 14.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5909288, upper bound: 1.5826375
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5884258, upper bound: 1.5851470
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2707267, 3.2655182
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4072480, 3.4034216
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8273392, 2.8248730
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7996569, 3.7962914
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2517738, 3.2459121
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6985192, 2.7051988
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5965867, 3.5929189
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4443240, 3.4431129
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9847717, 2.9823012
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4257748, 2.4229934

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5896650, upper bound: 1.5859510
time: 4.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5901741, upper bound: 1.5854397
time: 5.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2146387, 3.2014456
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4070134, 3.4025941
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8038273, 2.8056452
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7871923, 3.7917099
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2472849, 3.2398319
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6763649, 2.6787548
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5802660, 3.5730362
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4381185, 3.4324546
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8943515, 2.9036014
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4344420, 2.4306474

Time for backsubstitution: 15.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6238

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5811492, upper bound: 1.5842667
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5794714, upper bound: 1.5859478
time: 5.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2060747, 3.2100103
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4060121, 3.4035945
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8083782, 2.8010943
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7925253, 3.7863770
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2441483, 3.2429686
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6749859, 2.6801324
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5747919, 3.5785093
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4345508, 3.4360218
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9007850, 2.8971679
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4336057, 2.4314837

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5849083, upper bound: 1.5808841
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5851539, upper bound: 1.5806537
time: 4.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2746263, 3.2798748
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.3842754, 3.3867579
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8104043, 2.8153510
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7718010, 3.7739563
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2387767, 3.2421637
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6862502, 2.6821935
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5712543, 3.5776134
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4346104, 3.4366689
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9374442, 2.9349971
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4222617, 2.4244580

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5857587, upper bound: 1.5906466
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5849592, upper bound: 1.5914397
time: 4.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.2743077, 3.2801914
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.3844433, 3.3865895
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.8104157, 2.8153398
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7720070, 3.7737503
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2390723, 3.2418680
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6863275, 2.6821163
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.5714831, 3.5773840
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4349136, 3.4363651
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9370341, 2.9354062
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4219913, 2.4247279

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5815701, upper bound: 1.5912047
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5859901, upper bound: 1.5867830
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1358309, 3.1196675
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.3908949, 3.3992281
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7168689, 2.7043409
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7382412, 3.7340550
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2576208, 3.2606444
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6796541, 2.6783025
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.4168425, 3.3914189
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4414754, 3.4428658
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9026804, 2.8952317
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4198823, 2.4227126

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5820897, upper bound: 1.5916557
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5812957, upper bound: 1.5924497
time: 4.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1155510, 3.1399474
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.3922358, 3.3978891
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7032161, 2.7179940
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7334681, 3.7388272
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2556238, 3.2626405
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.6804295, 2.6775272
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3938875, 3.4143739
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4404302, 3.4439106
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.8980188, 2.8998933
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4199786, 2.4226158

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5864921, upper bound: 1.5855134
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5839784, upper bound: 1.5880243
time: 4.83 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5869634, upper bound: 1.5827714
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5861853, upper bound: 1.5835421
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5816786, upper bound: 1.5829957
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5815138, upper bound: 1.5884236
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5924339, upper bound: 1.5779565
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5869920, upper bound: 1.5834073
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5907443, upper bound: 1.5833945
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5890540, upper bound: 1.5850730
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5895516, upper bound: 1.5848644
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5899084, upper bound: 1.5845076
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5844628, upper bound: 1.5895083
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5836708, upper bound: 1.5903014
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5833092, upper bound: 1.5919958
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5838205, upper bound: 1.5914880
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5871502, upper bound: 1.5816169
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5822671, upper bound: 1.5865180
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5909288, upper bound: 1.5826375
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5884258, upper bound: 1.5851470
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5896650, upper bound: 1.5859510
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5901741, upper bound: 1.5854397
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5811492, upper bound: 1.5842667
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5794714, upper bound: 1.5859478
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5849083, upper bound: 1.5808841
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5851539, upper bound: 1.5806537
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5857587, upper bound: 1.5906466
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5849592, upper bound: 1.5914397
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5815701, upper bound: 1.5912047
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5859901, upper bound: 1.5867830
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5820897, upper bound: 1.5916557
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5812957, upper bound: 1.5924497
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5864921, upper bound: 1.5855134
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.17
Output dim: 9, lower bound: -1.5839784, upper bound: 1.5880243

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.1632252, 3.1407661
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.4420338, 3.4452491
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7528000, 2.7397118
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.7529588, 3.7497368
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.2924447, 3.2971969
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.7234383, 2.7202380
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.4178300, 3.3964753
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.4396586, 3.4349337
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.9273810, 2.9287758
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.4436698, 2.4423027

Time for backsubstitution: 14.66 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.44260835647583
rel_dist={9: [-1.592849651897651, 1.5928489343896768]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 471
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 471

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4146641, upper bound: 1.4147653
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147667, upper bound: 1.4146631
time: 4.58 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.25 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.25
Output dim: 9, lower bound: -1.4146641, upper bound: 1.4147653
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.25
Output dim: 9, lower bound: -1.4147667, upper bound: 1.4146631

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0841160, 3.0836287
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2073116, 3.2077603
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7276049, 2.7278230
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5795898, 3.5796318
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1632462, 3.1636953
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5671210, 2.5666337
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3321829, 3.3339596
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3060484, 3.3055382
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7221727, 2.7238259
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3676190, 2.3681645

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4131111, upper bound: 1.4147641
time: 4.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4146625, upper bound: 1.4132123
time: 5.11 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0836296, 3.0841155
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2077599, 3.2073121
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7278233, 2.7276046
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5796318, 3.5795898
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1636944, 3.1632462
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5666327, 2.5671210
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3339596, 3.3321824
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3055382, 3.3060489
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7238255, 2.7221730
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3681645, 2.3676190

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 568

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 80

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4139272, upper bound: 1.4124021
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4125063, upper bound: 1.4138267
time: 4.20 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.47 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.47
Output dim: 9, lower bound: -1.4131111, upper bound: 1.4147641
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.47
Output dim: 9, lower bound: -1.4146625, upper bound: 1.4132123
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.47
Output dim: 9, lower bound: -1.4139272, upper bound: 1.4124021
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.47
Output dim: 9, lower bound: -1.4125063, upper bound: 1.4138267

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0182557, 3.0149138
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2076392, 3.2077544
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7019296, 2.7036648
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5857921, 3.5876126
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1540890, 3.1534925
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5671000, 2.5661540
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2988124, 3.2987652
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3044600, 3.3027601
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6780696, 2.6818678
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3743110, 2.3745778

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6139

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4129662, upper bound: 1.4147643
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4131108, upper bound: 1.4146185
time: 4.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0154004, 3.0177686
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2073064, 3.2080877
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7034469, 2.7021480
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5875697, 3.5858350
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1530437, 3.1545382
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5666404, 2.5666132
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2969880, 3.3005896
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3032708, 3.3039494
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6802144, 2.6797233
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3740325, 2.3748567

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4146620, upper bound: 1.4119835
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4134344, upper bound: 1.4132128
time: 4.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0836248, 3.0841112
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2077570, 3.2073083
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7278242, 2.7276053
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5796223, 3.5795798
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1636868, 3.1632390
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5666380, 2.5671248
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3339710, 3.3321929
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3055391, 3.3060498
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7238317, 2.7221777
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3681602, 2.3676152

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4139267, upper bound: 1.4111235
time: 6.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4126412, upper bound: 1.4124012
time: 4.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0836248, 3.0841112
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2077570, 3.2073083
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7278242, 2.7276053
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5796223, 3.5795789
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1636887, 3.1632390
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5666370, 2.5671253
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3339710, 3.3321934
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3055391, 3.3060498
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7238307, 2.7221782
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3681602, 2.3676147

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4122000, upper bound: 1.4136156
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4122524, upper bound: 1.4136198
time: 4.98 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.28 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4129662, upper bound: 1.4147643
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4131108, upper bound: 1.4146185
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4146620, upper bound: 1.4119835
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4134344, upper bound: 1.4132128
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4139267, upper bound: 1.4111235
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4126412, upper bound: 1.4124012
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4122000, upper bound: 1.4136156
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4122524, upper bound: 1.4136198

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0035458, 3.0008163
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1991267, 3.1988740
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6824179, 2.6859608
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5948362, 3.5975966
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1496921, 3.1494946
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5603013, 2.5590582
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2959185, 3.2973261
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3104687, 3.3085165
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6756964, 2.6792939
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3824215, 2.3822486

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6238

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4129657, upper bound: 1.4141775
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4123800, upper bound: 1.4147631
time: 4.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0041590, 3.0002036
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1987586, 3.1992421
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6842260, 2.6841536
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5957775, 3.5966563
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1500907, 3.1490951
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5600057, 2.5593538
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2973728, 3.2958713
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3102169, 3.3087683
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6754971, 2.6794944
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3819818, 2.3826888

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 568

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6238

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4131103, upper bound: 1.4140323
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4125246, upper bound: 1.4146179
time: 4.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0146761, 3.0161955
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2060170, 3.2074947
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7027488, 2.7018259
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5874233, 3.5855060
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1499882, 3.1531277
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5664382, 2.5661621
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2969222, 3.3004465
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3024979, 3.3022437
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6795096, 2.6794109
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3735590, 2.3738348

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 485

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4131469, upper bound: 1.4119821
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4146611, upper bound: 1.4104691
time: 4.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0138273, 3.0170436
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2067132, 3.2067990
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7031245, 2.7014499
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5872412, 3.5856881
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1516323, 3.1514826
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5661902, 2.5664105
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2968440, 3.3005242
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3015652, 3.3031764
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6799026, 2.6790185
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3730106, 2.3743832

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 485

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4119193, upper bound: 1.4132109
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4134336, upper bound: 1.4116973
time: 4.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0829000, 3.0825396
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2064686, 3.2067158
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7271261, 2.7272830
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5794730, 3.5792499
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1606331, 3.1618290
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5664349, 2.5666738
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3339043, 3.3320489
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3047657, 3.3043437
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7231259, 2.7218647
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3676867, 2.3665934

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6165

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4123699, upper bound: 1.4111218
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4139251, upper bound: 1.4095667
time: 4.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0856504, 3.0860310
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2077284, 3.2073359
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7279205, 2.7277052
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5808744, 3.5809002
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1647220, 3.1643715
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5660110, 2.5664737
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3320723, 3.3303723
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3038850, 3.3042965
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7204123, 2.7188938
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3659177, 2.3654573

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4121993, upper bound: 1.4123299
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4109203, upper bound: 1.4136155
time: 5.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0855436, 3.0861380
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2077847, 3.2072802
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7279263, 2.7277017
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5809450, 3.5808315
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1648211, 3.1642733
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5660377, 2.5664995
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3321514, 3.3302960
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3039861, 3.3043962
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7205458, 2.7190301
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3660035, 2.3655474

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 568

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4122521, upper bound: 1.4133955
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4120277, upper bound: 1.4136194
time: 4.54 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 9, lower bound: -1.4129657, upper bound: 1.4141775
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 9, lower bound: -1.4123800, upper bound: 1.4147631
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 9, lower bound: -1.4131103, upper bound: 1.4140323
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 9, lower bound: -1.4125246, upper bound: 1.4146179
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.93
Output dim: 9, lower bound: -1.4131469, upper bound: 1.4119821
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 9, lower bound: -1.4146611, upper bound: 1.4104691
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 9, lower bound: -1.4119193, upper bound: 1.4132109
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 9, lower bound: -1.4134336, upper bound: 1.4116973
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.93
Output dim: 9, lower bound: -1.4123699, upper bound: 1.4111218
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 9, lower bound: -1.4139251, upper bound: 1.4095667
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.93
Output dim: 9, lower bound: -1.4121993, upper bound: 1.4123299
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 9, lower bound: -1.4109203, upper bound: 1.4136155
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 9, lower bound: -1.4122521, upper bound: 1.4133955
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.93
Output dim: 9, lower bound: -1.4120277, upper bound: 1.4136194

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9817867, 2.9804709
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1993694, 3.1988583
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6625051, 2.6673164
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6013207, 3.6036754
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1471081, 3.1466269
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5517635, 2.5508726
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2526112, 3.2558594
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3106613, 3.3084989
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6751609, 2.6785080
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3733120, 2.3727043

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4129653, upper bound: 1.4138400
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4126277, upper bound: 1.4141769
time: 5.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9832001, 2.9790585
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1991119, 3.1991158
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6637745, 2.6660473
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6009145, 3.6040826
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1468239, 3.1469107
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5521154, 2.5505207
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2544527, 3.2540183
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3104506, 3.3087091
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6749110, 2.6787581
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3728771, 2.3731391

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4123792, upper bound: 1.4142295
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4118460, upper bound: 1.4147628
time: 5.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9824009, 2.9798579
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1990013, 3.1992264
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6643124, 2.6655095
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6022630, 3.6027350
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1475077, 3.1462274
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5514679, 2.5511682
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2540655, 3.2544045
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3104095, 3.3087506
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6749606, 2.6787086
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3728724, 2.3731444

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 485

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4131096, upper bound: 1.4134985
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4125764, upper bound: 1.4140314
time: 5.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9838123, 2.9784458
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1987438, 3.1994843
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6655817, 2.6642401
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6018567, 3.6031413
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1472235, 3.1465111
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5518198, 2.5508163
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2559061, 3.2525635
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3101988, 3.3089609
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6747108, 2.6789584
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3724365, 2.3735793

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4125241, upper bound: 1.4133895
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4112963, upper bound: 1.4146173
time: 5.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.8524303, 2.8607099
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2140760, 3.2151072
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5934563, 2.5970845
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5491962, 3.5488701
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1651893, 3.1689954
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5604768, 2.5599430
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1132789, 3.1244545
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3104377, 3.3105316
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6421452, 2.6436005
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3728154, 2.3730593

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4146607, upper bound: 1.4101386
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4143238, upper bound: 1.4104687
time: 4.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.8583422, 2.8547978
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2143259, 3.2148578
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5983829, 2.5921576
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5506058, 3.5474606
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1675010, 3.1666851
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5599713, 2.5604498
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1208529, 3.1168804
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3098540, 3.3111157
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6440926, 2.6416543
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3722346, 2.3736401

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6126

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4117056, upper bound: 1.4129516
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4117021, upper bound: 1.4129034
time: 4.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.8515816, 2.8615577
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2147722, 3.2144115
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.5938320, 2.5967085
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5490141, 3.5490513
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1668353, 3.1673503
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5602288, 2.5601914
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.1132016, 3.1245317
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3095050, 3.3114638
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6425381, 2.6432083
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3722670, 2.3736076

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6126

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4134328, upper bound: 1.4111634
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4128998, upper bound: 1.4116970
time: 4.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0141859, 3.0166802
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2064633, 3.2070446
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7029672, 2.7016079
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5874534, 3.5854530
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1504297, 3.1526718
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5659566, 2.5666542
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2987094, 3.2986784
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3019876, 3.3027549
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6811695, 2.6777630
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3741007, 2.3732853

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 568

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4139247, upper bound: 1.4093420
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4137003, upper bound: 1.4095665
time: 4.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0840816, 3.0853109
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2071381, 3.2060497
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7275982, 2.7270069
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5805407, 3.5807476
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1633162, 3.1613197
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5655594, 2.5662692
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.3319273, 3.3303051
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3021750, 3.3035183
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7200928, 2.7181816
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3648963, 2.3649845

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6126

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4109195, upper bound: 1.4130836
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4103863, upper bound: 1.4136166
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0929923, 3.0938993
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1898651, 3.1885810
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7161574, 2.7166662
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5717678, 3.5720367
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1553473, 3.1543875
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5697098, 2.5698380
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2971578, 3.2967620
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3013577, 3.3016534
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7185612, 2.7173054
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3622282, 2.3616095

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6165
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 6238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4122514, upper bound: 1.4116370
time: 11.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4119172, upper bound: 1.4133973
time: 4.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -3.0933051, 3.0935862
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1890850, 3.1893606
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.7168899, 2.7159331
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.5721493, 3.5716543
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1549354, 3.1547995
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5693750, 2.5701723
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2986169, 3.2953029
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3012443, 3.3017673
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7188215, 2.7170446
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3620651, 2.3617730

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 6238
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6139
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5749

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4120269, upper bound: 1.4123333
time: 7.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4107482, upper bound: 1.4136217
time: 4.90 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4129653, upper bound: 1.4138400
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4126277, upper bound: 1.4141769
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4123792, upper bound: 1.4142295
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4118460, upper bound: 1.4147628
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4131096, upper bound: 1.4134985
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4125764, upper bound: 1.4140314
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4125241, upper bound: 1.4133895
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4112963, upper bound: 1.4146173
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4146607, upper bound: 1.4101386
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4143238, upper bound: 1.4104687
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4117056, upper bound: 1.4129516
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4117021, upper bound: 1.4129034
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4134328, upper bound: 1.4111634
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4128998, upper bound: 1.4116970
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4139247, upper bound: 1.4093420
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4137003, upper bound: 1.4095665
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4109195, upper bound: 1.4130836
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4103863, upper bound: 1.4136166
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4122514, upper bound: 1.4116370
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4119172, upper bound: 1.4133973
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4120269, upper bound: 1.4123333
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.80
Output dim: 9, lower bound: -1.4107482, upper bound: 1.4136217

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9771433, 2.9760191
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2001934, 3.1998174
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6604304, 2.6651521
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6174359, 3.6206408
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1429868, 3.1430206
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5757866, 2.5739260
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2372942, 3.2411790
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3179541, 3.3154964
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7162123, 2.7213230
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3719523, 2.3712857

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 66

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4127563, upper bound: 1.4135759
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4127518, upper bound: 1.4135273
time: 4.57 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9773350, 2.9758267
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.2003288, 3.1996815
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6603408, 2.6652412
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6182866, 3.6197901
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1435018, 3.1425052
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5748158, 2.5748963
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2379293, 3.2405419
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3176584, 3.3157916
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.7179747, 2.7195599
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3718932, 2.3713443

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6126
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6126

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4126269, upper bound: 1.4136431
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4120937, upper bound: 1.4141768
time: 4.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9810114, 2.9756756
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1987486, 3.1985555
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6637502, 2.6660106
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6007428, 3.6039720
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1382437, 3.1413574
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5484066, 2.5481188
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2532530, 3.2532420
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3093290, 3.3069749
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6745086, 2.6784987
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3706100, 2.3696368

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 163

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4122200, upper bound: 1.4139718
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4121227, upper bound: 1.4140694
time: 4.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9798174, 2.9768691
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1985502, 3.1987524
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6637378, 2.6660228
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6008039, 3.6039109
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1412706, 3.1383309
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5497131, 2.5468123
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2536764, 3.2528186
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3087168, 3.3075891
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6746516, 2.6783566
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3693745, 2.3708727

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 6208

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6155

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4118456, upper bound: 1.4144248
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4115077, upper bound: 1.4147621
time: 4.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9802103, 2.9764750
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1986380, 3.1986661
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6642871, 2.6654727
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6020913, 3.6026235
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1389275, 3.1406741
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5477591, 2.5487664
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2528658, 3.2536283
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3092890, 3.3070164
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6745591, 2.6784489
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3706052, 2.3696420

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 5749
type: RSZ, layer: 1, pos: 163

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4115948, upper bound: 1.4134982
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4131087, upper bound: 1.4119849
time: 4.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9790182, 2.9776683
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1984396, 3.1988635
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6642756, 2.6654849
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6021523, 3.6025624
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1419535, 3.1376476
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5490656, 2.5474598
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2532892, 3.2532048
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3086748, 3.3076305
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6747012, 2.6783071
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3693697, 2.3708775

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6208
type: RSZ, layer: 1, pos: 568
type: RSZ, layer: 1, pos: 6155
type: RSZ, layer: 1, pos: 66
type: RSZ, layer: 1, pos: 163
type: RSZ, layer: 1, pos: 485
type: RSZ, layer: 1, pos: 80
type: RSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6208

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4110616, upper bound: 1.4140313
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4125755, upper bound: 1.4125170
time: 4.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.6942806, -5.0616770, -9.6942806, -5.0616770, -2.9830885, 2.9768736
1: -15.0952425, -10.8431473, -15.0952425, -10.8431473, -3.1974554, 3.1988921
2: -9.0615978, -5.7651587, -9.0615978, -5.7651587, -2.6648827, 2.6639178
3: -11.5230656, -7.4036608, -11.5230656, -7.4036608, -3.6017094, 3.6028132
4: -5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.1441679, 3.1451020
5: -3.5736499, -0.4953117, -3.5736499, -0.4953117, -2.5516176, 2.5503652
6: -11.5837259, -6.9704914, -11.5837259, -6.9704914, -3.2558403, 3.2524204
7: -2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.3094263, 3.3072562
8: -5.0775828, -1.4732656, -5.0775828, -1.4732656, -2.6740050, 2.6786461
9: 0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.3719645, 2.3725584

Time for backsubstitution: 14.69 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.3590686321258545
rel_dist={9: [-1.414766908177059, 1.4147664541724545]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 2414.30 seconds
