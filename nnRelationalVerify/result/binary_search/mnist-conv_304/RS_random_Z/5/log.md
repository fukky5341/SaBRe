## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.3532293525
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.8783393, 3.8783391)
1: (-12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.5584154, 3.5584154)
2: (-13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.2301512, 3.2301512)
3: (-9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981)
4: (-4.5608406, -2.3997998, -4.5608406, -2.3997998, -2.1610408, 2.1610408)
5: (-11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.7072897, 3.7072897)
6: (-17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.9770737, 3.9770737)
7: (-6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.8377733, 2.8377733)
8: (-2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.2236829, 2.2236829)
9: (2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.7430749, 2.7430749)

## BASE Result
execution time: IAR + LP analysis = 15.03 + 34.63 = 49.66 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -2.1375822, upper bound: 2.1375790


# Binary Search by BASE starts (time budget: 3550.34 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.467238664627075
rel_dist={9: [-1.6640502761084588, 1.6640497405138106]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.2999587059020996
rel_dist={9: [-1.360028225390102, 1.3600276268876046]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.188438653945923
rel_dist={9: [-1.0985895441039362, 1.0985873760165363]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.244198799133301
rel_dist={9: [-1.2428127078072388, 1.2428113778761434]}

## Binary Search Result
Binary search time: 211.35 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3339.00 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 6222
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7532678, upper bound: 1.7318902
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7318886, upper bound: 1.7532674
time: 6.42 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 10.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 10.57
Output dim: 9, lower bound: -1.7532678, upper bound: 1.7318902
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 10.57
Output dim: 9, lower bound: -1.7318886, upper bound: 1.7532674

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1305161, 3.1313291
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0340509, 3.0360312
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0236273, 3.0254469
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8642249, 1.8683381
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0875182, 3.0880985
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3723249, 3.3737168
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5076919, 2.5049424
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0371962, 2.0360935
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5181077, 2.5160935

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 6222
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7518363, upper bound: 1.7318838
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7532631, upper bound: 1.7304416
time: 4.04 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1313295, 3.1305161
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0360308, 3.0340500
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0254469, 3.0236273
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8683381, 1.8642247
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0880980, 3.0875177
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3737164, 3.3723247
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5049424, 2.5076916
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0360937, 2.0371964
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5160935, 2.5181079

Time for backsubstitution: 12.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 6222
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7304410, upper bound: 1.7531980
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7318137, upper bound: 1.7518379
time: 4.56 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.94 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.94
Output dim: 9, lower bound: -1.7518363, upper bound: 1.7318838
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.94
Output dim: 9, lower bound: -1.7532631, upper bound: 1.7304416
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.94
Output dim: 9, lower bound: -1.7304410, upper bound: 1.7531980
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.94
Output dim: 9, lower bound: -1.7318137, upper bound: 1.7518379

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1235147, 3.1110954
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0305777, 3.0260816
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0168085, 3.0230789
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8596778, 1.8551829
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0796428, 3.0853591
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3647795, 3.3519340
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5075750, 2.5046229
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0356784, 2.0317175
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5131543, 2.5143814

Time for backsubstitution: 13.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 6222
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7518266, upper bound: 1.7318084
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7517235, upper bound: 1.7304353
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1102824, 3.1243272
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0241003, 3.0325594
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0212593, 3.0186276
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8510690, 1.8637912
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0847774, 3.0802240
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3505421, 3.3661716
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5073724, 2.5048256
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0328202, 2.0345752
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5163958, 2.5111401

Time for backsubstitution: 12.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7508167, upper bound: 1.7304392
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7532607, upper bound: 1.7279972
time: 4.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1317964, 3.1274734
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0361781, 3.0329561
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0226011, 3.0240760
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8685713, 1.8627400
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0872884, 3.0876322
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3742867, 3.3685892
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5050058, 2.5072405
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0361910, 2.0365226
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5147727, 2.5183096

Time for backsubstitution: 13.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7297168, upper bound: 1.7310917
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7297028, upper bound: 1.7531598
time: 5.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1282868, 3.1305161
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0349364, 3.0340500
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0254469, 3.0207825
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8668537, 1.8642247
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0880980, 3.0867076
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3699818, 3.3723247
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5044913, 2.5076916
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0354195, 2.0371964
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5160935, 2.5167868

Time for backsubstitution: 12.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7293694, upper bound: 1.7518329
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7318113, upper bound: 1.7493888
time: 5.18 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.81 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.81
Output dim: 9, lower bound: -1.7518266, upper bound: 1.7318084
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.81
Output dim: 9, lower bound: -1.7517235, upper bound: 1.7304353
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.81
Output dim: 9, lower bound: -1.7508167, upper bound: 1.7304392
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.81
Output dim: 9, lower bound: -1.7532607, upper bound: 1.7279972
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.81
Output dim: 9, lower bound: -1.7297168, upper bound: 1.7310917
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.81
Output dim: 9, lower bound: -1.7297028, upper bound: 1.7531598
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.81
Output dim: 9, lower bound: -1.7293694, upper bound: 1.7518329
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.81
Output dim: 9, lower bound: -1.7318113, upper bound: 1.7493888

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1239796, 3.1080523
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0307426, 3.0249877
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0139627, 3.0235281
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8599100, 1.8536978
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0788302, 3.0854826
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3653498, 3.3481994
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5076442, 2.5041718
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0357804, 2.0310442
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5118334, 2.5145831

Time for backsubstitution: 12.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7517911, upper bound: 1.7310701
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7297105, upper bound: 1.7310840
time: 4.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1204720, 3.1110954
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0294838, 3.0260816
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0168085, 3.0202341
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8581924, 1.8551829
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0796428, 3.0845470
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3610449, 3.3519340
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5071235, 2.5046229
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0350051, 2.0317175
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5131543, 2.5130603

Time for backsubstitution: 12.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7516880, upper bound: 1.7296997
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7296112, upper bound: 1.7297110
time: 5.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1047521, 3.1232028
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0214262, 3.0320086
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0051336, 3.0153499
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9849420
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8460264, 1.8627672
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0806174, 3.0598669
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3481436, 3.3656831
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5014606, 2.5036132
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0311337, 2.0263546
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5130975, 2.5104661

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7493840, upper bound: 1.7303284
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7507441, upper bound: 1.7304291
time: 5.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1091571, 3.1187973
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0235500, 3.0298848
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0179815, 3.0025020
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9869847, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8500462, 1.8587484
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0644202, 3.0760665
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3500528, 3.3637736
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5061603, 2.4989140
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0246000, 2.0328896
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5157216, 2.5078418

Time for backsubstitution: 13.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7532252, upper bound: 1.7272591
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7311601, upper bound: 1.7272731
time: 4.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1368442, 3.1351752
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0039635, 3.0101414
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0412340, 3.0502930
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8174171, 1.8271334
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0965366, 3.0996180
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3565111, 3.3559988
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5104828, 2.5035453
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0236158, 2.0187719
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4968386, 2.4929948

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7272725, upper bound: 1.7310892
time: 7.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7297144, upper bound: 1.7286474
time: 6.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1394973, 3.1325202
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0133629, 3.0007429
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0488234, 3.0427084
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8329639, 1.8115857
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0992737, 3.0968800
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3616953, 3.3508146
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5013113, 2.5127299
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0184402, 2.0239475
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4894576, 2.5003760

Time for backsubstitution: 12.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7296937, upper bound: 1.7531548
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7296975, upper bound: 1.7516878
time: 4.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1227565, 3.1293912
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0322633, 3.0334997
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0093203, 3.0175037
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9871616
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8618107, 1.8632016
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0839410, 3.0663505
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3675833, 3.3718362
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.4985800, 2.5064793
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0337324, 2.0289752
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5127957, 2.5161138

Time for backsubstitution: 12.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7286451, upper bound: 1.7297194
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7286312, upper bound: 1.7517977
time: 4.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1271615, 3.1249857
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0343862, 3.0313759
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0221682, 3.0046558
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9854379, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8658304, 1.8591819
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0677419, 3.0825500
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3694925, 3.3699265
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5032787, 2.5017800
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0271988, 2.0355089
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5154202, 2.5134892

Time for backsubstitution: 12.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7303285, upper bound: 1.7493862
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7318060, upper bound: 1.7493801
time: 4.38 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.84
Output dim: 9, lower bound: -1.7517911, upper bound: 1.7310701
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.84
Output dim: 9, lower bound: -1.7297105, upper bound: 1.7310840
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.84
Output dim: 9, lower bound: -1.7516880, upper bound: 1.7296997
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.84
Output dim: 9, lower bound: -1.7296112, upper bound: 1.7297110
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.84
Output dim: 9, lower bound: -1.7493840, upper bound: 1.7303284
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.84
Output dim: 9, lower bound: -1.7507441, upper bound: 1.7304291
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.84
Output dim: 9, lower bound: -1.7532252, upper bound: 1.7272591
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.84
Output dim: 9, lower bound: -1.7311601, upper bound: 1.7272731
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.84
Output dim: 9, lower bound: -1.7272725, upper bound: 1.7310892
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.84
Output dim: 9, lower bound: -1.7297144, upper bound: 1.7286474
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.84
Output dim: 9, lower bound: -1.7296937, upper bound: 1.7531548
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.84
Output dim: 9, lower bound: -1.7296975, upper bound: 1.7516878
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.84
Output dim: 9, lower bound: -1.7286451, upper bound: 1.7297194
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.84
Output dim: 9, lower bound: -1.7286312, upper bound: 1.7517977
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.84
Output dim: 9, lower bound: -1.7303285, upper bound: 1.7493862
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.84
Output dim: 9, lower bound: -1.7318060, upper bound: 1.7493801

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1290283, 3.1157541
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9985294, 3.0021725
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0325947, 3.0497489
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8087564, 1.8180907
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0880795, 3.0974693
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3475747, 3.3356080
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5131345, 2.5004766
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0232053, 2.0132935
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4938998, 2.4892688

Time for backsubstitution: 12.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7493446, upper bound: 1.7310677
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7517887, upper bound: 1.7286260
time: 4.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1316824, 3.1130996
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0079279, 2.9927750
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0401802, 3.0421596
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8243041, 1.8025441
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0908175, 3.0947313
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3527589, 3.3304238
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5039496, 2.5096488
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0180297, 2.0184691
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4865189, 2.4966497

Time for backsubstitution: 12.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7272661, upper bound: 1.7310816
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7297081, upper bound: 1.7286397
time: 4.64 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1255188, 3.1187978
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9972715, 3.0032663
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0354404, 3.0464554
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8070388, 1.8195760
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0888910, 3.0965338
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3432698, 3.3393433
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5126138, 2.5009282
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0224299, 2.0139663
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4952211, 2.4877460

Time for backsubstitution: 12.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7491902, upper bound: 1.7296947
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7516839, upper bound: 1.7272555
time: 3.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1281738, 3.1161432
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0066700, 2.9938684
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0430250, 3.0388656
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8225865, 1.8040292
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0916290, 3.0937958
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3484540, 3.3341594
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5034289, 2.5101004
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0172544, 2.0191419
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4878402, 2.4951270

Time for backsubstitution: 12.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7271171, upper bound: 1.7297088
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7296071, upper bound: 1.7272668
time: 5.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1052189, 3.1201591
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0215731, 3.0309153
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0022879, 3.0157981
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9849334
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8462596, 1.8612821
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0798059, 3.0599794
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3487139, 3.3619490
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5015244, 2.5031619
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0312309, 2.0256813
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5117757, 2.5106671

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7493485, upper bound: 1.7295905
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7272700, upper bound: 1.7296064
time: 4.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1017075, 3.1232028
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0203323, 3.0320086
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0051336, 3.0125046
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9849420
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8445411, 1.8627672
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0806174, 3.0590549
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3444090, 3.3656831
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5010095, 2.5036132
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0304604, 2.0263546
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5130975, 2.5091443

Time for backsubstitution: 12.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7507086, upper bound: 1.7296913
time: 4.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7286427, upper bound: 1.7297049
time: 4.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1142054, 3.1264997
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9913368, 3.0070701
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0366144, 3.0287232
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.7988925, 1.8231413
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0736694, 3.0880537
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3322792, 3.3511829
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5116491, 2.4952190
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0120249, 2.0151386
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4977875, 2.4825268

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7517925, upper bound: 1.7271006
time: 4.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7531526, upper bound: 1.7272492
time: 4.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1168604, 3.1238446
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0007362, 2.9976721
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0441990, 3.0211339
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8144403, 1.8075948
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0764074, 3.0853152
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3374634, 3.3459988
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5024652, 2.5043910
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0068493, 2.0203140
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4904070, 2.4899077

Time for backsubstitution: 12.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7297119, upper bound: 1.7271141
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7310845, upper bound: 1.7272633
time: 5.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1313128, 3.1340504
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0012903, 3.0095911
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0251064, 3.0470142
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8123746, 1.8261104
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0923786, 3.0792613
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3541136, 3.3555102
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5045710, 2.5023322
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0219288, 2.0105512
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4935417, 2.4923222

Time for backsubstitution: 13.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7272632, upper bound: 1.7310843
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7272672, upper bound: 1.7296070
time: 4.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1357179, 3.1296449
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0034142, 3.0074673
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0379553, 3.0341663
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8163943, 1.8220906
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0761795, 3.0954614
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3560228, 3.3536005
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5092697, 2.4976330
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0153952, 2.0170848
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4961662, 2.4896977

Time for backsubstitution: 13.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7297052, upper bound: 1.7286424
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7297090, upper bound: 1.7271170
time: 4.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1324959, 3.1122866
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0099077, 2.9907937
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0420036, 3.0403404
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8284168, 1.7984307
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0913982, 3.0941505
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3541503, 3.3290319
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5012002, 2.5124109
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0169272, 2.0195720
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4845047, 2.4986641

Time for backsubstitution: 12.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7272493, upper bound: 1.7531525
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7296913, upper bound: 1.7507085
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1192636, 3.1255183
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0034132, 2.9972715
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0464554, 3.0358882
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8198085, 1.8070390
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0965338, 3.0890040
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3399129, 3.3432698
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5009913, 2.5126133
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0140653, 2.0224297
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4877458, 2.4954228

Time for backsubstitution: 13.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7272533, upper bound: 1.7516838
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7296951, upper bound: 1.7491899
time: 4.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1278033, 3.1370940
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0000496, 3.0106854
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0279522, 3.0437207
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8106570, 1.8275957
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0931911, 3.0783367
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3498087, 3.3592443
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5040560, 2.5027838
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0211573, 2.0112245
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4948626, 2.4907994

Time for backsubstitution: 12.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7271143, upper bound: 1.7297116
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7286398, upper bound: 1.7297080
time: 5.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1304584, 3.1344390
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0094471, 3.0012870
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0355415, 3.0361357
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8262029, 1.8120480
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0959291, 3.0755987
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3549929, 3.3540602
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.4948835, 2.5119684
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0159817, 2.0163999
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4874811, 2.4981804

Time for backsubstitution: 12.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7271004, upper bound: 1.7517921
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7286259, upper bound: 1.7517883
time: 4.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1201591, 3.1047521
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0309153, 3.0214262
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0153494, 3.0022888
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9849343, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8612819, 1.8460267
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0598674, 3.0798054
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3619490, 3.3481438
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5031619, 2.5014608
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0256815, 2.0311334
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5104659, 2.5117757

Time for backsubstitution: 12.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7296042, upper bound: 1.7272697
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7295903, upper bound: 1.7493481
time: 4.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1069279, 3.1179838
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0244379, 3.0279040
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0198011, 2.9978371
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9852605, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8526750, 1.8546350
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0650020, 3.0746737
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3477106, 3.3623817
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5029597, 2.5016634
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0228233, 2.0339925
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.5137074, 2.5085342

Time for backsubstitution: 12.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7310817, upper bound: 1.7272656
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7310678, upper bound: 1.7493441
time: 5.46 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.25 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7493446, upper bound: 1.7310677
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7517887, upper bound: 1.7286260
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7272661, upper bound: 1.7310816
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7297081, upper bound: 1.7286397
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7491902, upper bound: 1.7296947
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7516839, upper bound: 1.7272555
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7271171, upper bound: 1.7297088
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7296071, upper bound: 1.7272668
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7493485, upper bound: 1.7295905
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7272700, upper bound: 1.7296064
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7507086, upper bound: 1.7296913
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7286427, upper bound: 1.7297049
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7517925, upper bound: 1.7271006
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7531526, upper bound: 1.7272492
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7297119, upper bound: 1.7271141
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7310845, upper bound: 1.7272633
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7272632, upper bound: 1.7310843
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7272672, upper bound: 1.7296070
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7297052, upper bound: 1.7286424
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7297090, upper bound: 1.7271170
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7272493, upper bound: 1.7531525
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7296913, upper bound: 1.7507085
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7272533, upper bound: 1.7516838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7296951, upper bound: 1.7491899
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7271143, upper bound: 1.7297116
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7286398, upper bound: 1.7297080
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7271004, upper bound: 1.7517921
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7286259, upper bound: 1.7517883
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7296042, upper bound: 1.7272697
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7295903, upper bound: 1.7493481
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7310817, upper bound: 1.7272656
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.25
Output dim: 9, lower bound: -1.7310678, upper bound: 1.7493441

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1234970, 3.1146293
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9958544, 3.0016222
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0164690, 3.0464706
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8037133, 1.8170676
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0839224, 3.0771132
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3451762, 3.3351192
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5072217, 2.4992638
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0215192, 2.0050724
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4906011, 2.4885941

Time for backsubstitution: 12.70 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 2222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 975

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7476021, upper bound: 1.7309752
time: 4.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7491912, upper bound: 1.7293432
time: 6.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1279030, 3.1102238
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.9979792, 2.9994984
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0293169, 3.0336232
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8077321, 1.8130479
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0677223, 3.0933099
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3470855, 3.3332098
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.5119214, 2.4945648
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0149846, 2.0116057
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4932251, 2.4859698

Time for backsubstitution: 12.86 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2578

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2516

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6629791, upper bound: 1.6621474
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.6629791, upper bound: 1.6621474
time: 4.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -3.1261520, 3.1119742
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -3.0052538, 2.9922242
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -3.0240536, 3.0388813
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.9876981, 2.9876981
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.8192611, 1.8015208
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -3.0866604, 3.0743747
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -3.3503604, 3.3299353
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.4980369, 2.5084357
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -2.0163436, 2.0102477
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.4832201, 2.4959750

Time for backsubstitution: 12.75 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 2536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2615

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7045467, upper bound: 1.7090450
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.7053928, upper bound: 1.7082020
time: 5.35 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 22.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.55
Output dim: 9, lower bound: -1.7476021, upper bound: 1.7309752
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.55
Output dim: 9, lower bound: -1.7491912, upper bound: 1.7293432
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.55
Output dim: 9, lower bound: -1.6629791, upper bound: 1.6621474
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.55
Output dim: 9, lower bound: -1.6629791, upper bound: 1.6621474
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 22.55
Output dim: 9, lower bound: -1.7045467, upper bound: 1.7090450
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 22.55
Output dim: 9, lower bound: -1.7053928, upper bound: 1.7082020
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7297081, upper bound: 1.7286397
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7491902, upper bound: 1.7296947
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7516839, upper bound: 1.7272555
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7271171, upper bound: 1.7297088
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7296071, upper bound: 1.7272668
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7493485, upper bound: 1.7295905
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7272700, upper bound: 1.7296064
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7507086, upper bound: 1.7296913
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7286427, upper bound: 1.7297049
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7517925, upper bound: 1.7271006
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7531526, upper bound: 1.7272492
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7297119, upper bound: 1.7271141
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7310845, upper bound: 1.7272633
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7272632, upper bound: 1.7310843
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7272672, upper bound: 1.7296070
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7297052, upper bound: 1.7286424
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7297090, upper bound: 1.7271170
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7272493, upper bound: 1.7531525
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7296913, upper bound: 1.7507085
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7272533, upper bound: 1.7516838
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7296951, upper bound: 1.7491899
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7271143, upper bound: 1.7297116
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7286398, upper bound: 1.7297080
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7271004, upper bound: 1.7517921
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7286259, upper bound: 1.7517883
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7296042, upper bound: 1.7272697
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7295903, upper bound: 1.7493481
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7310817, upper bound: 1.7272656
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.55
Output dim: 9, lower bound: -1.7310678, upper bound: 1.7493441
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.522998809814453
rel_dist={9: [-1.7535540543698414, 1.753553758870325]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4665492, upper bound: 1.4681588
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4681588, upper bound: 1.4665493
time: 4.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.48 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.48
Output dim: 9, lower bound: -1.4665492, upper bound: 1.4681588
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.48
Output dim: 9, lower bound: -1.4681588, upper bound: 1.4665493

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6773000, 2.6798167
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6798162, 2.6810293
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6505027, 2.6578441
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7774453, 2.7722373
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6586568, 1.6609538
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6813173, 2.6720610
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9907227, 2.9918137
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2664719, 2.2691572
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8355799, 1.8318465
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3524203, 2.3539202

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662636, upper bound: 1.4559389
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4543329, upper bound: 1.4678707
time: 5.47 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6798177, 2.6772990
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6810293, 2.6798158
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6578441, 2.6505022
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7722363, 2.7774458
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6609538, 1.6586568
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6720600, 2.6813178
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9918137, 2.9907222
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2691569, 2.2664719
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8318462, 1.8355799
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3539200, 2.3524208

Time for backsubstitution: 12.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4671955, upper bound: 1.4665437
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4681536, upper bound: 1.4655869
time: 4.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.96 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.96
Output dim: 9, lower bound: -1.4662636, upper bound: 1.4559389
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.96
Output dim: 9, lower bound: -1.4543329, upper bound: 1.4678707
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.96
Output dim: 9, lower bound: -1.4671955, upper bound: 1.4665437
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.96
Output dim: 9, lower bound: -1.4681536, upper bound: 1.4655869

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6766691, 2.6796513
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6730280, 2.6753736
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6490917, 2.6574736
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7829957, 2.7768064
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6445723, 1.6492202
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6828589, 2.6739349
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9859447, 2.9878306
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2585759, 2.2596896
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8324280, 1.8280640
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3466671, 2.3470154

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662434, upper bound: 1.4552019
time: 4.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4536033, upper bound: 1.4552093
time: 4.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6771336, 2.6791868
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6741600, 2.6742415
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6501322, 2.6564336
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7820144, 2.7777872
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6469231, 1.6468697
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6831918, 2.6736031
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9867401, 2.9870353
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2570043, 2.2612607
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8317976, 1.8286941
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3455160, 2.3481665

Time for backsubstitution: 12.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4533675, upper bound: 1.4678676
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4533699, upper bound: 1.4669103
time: 5.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6787786, 2.6742558
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6806450, 2.6787224
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6549983, 2.6495390
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7722344, 2.7774377
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6604507, 1.6571717
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6712494, 2.6810346
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9905376, 2.9869871
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2690010, 2.2660213
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8316135, 1.8349061
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3525977, 2.3519681

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4669099, upper bound: 1.4543252
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4549768, upper bound: 1.4662596
time: 4.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6767731, 2.6762614
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6799364, 2.6794314
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6568809, 2.6476569
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7722297, 2.7774429
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6594684, 1.6581533
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6717777, 2.6805062
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9880781, 2.9894476
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2687068, 2.2663155
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8311729, 1.8353469
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3534679, 2.3510981

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4678653, upper bound: 1.4533667
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4559335, upper bound: 1.4653010
time: 4.56 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 22.03 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.03
Output dim: 9, lower bound: -1.4662434, upper bound: 1.4552019
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.03
Output dim: 9, lower bound: -1.4536033, upper bound: 1.4552093
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.03
Output dim: 9, lower bound: -1.4533675, upper bound: 1.4678676
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.03
Output dim: 9, lower bound: -1.4533699, upper bound: 1.4669103
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.03
Output dim: 9, lower bound: -1.4669099, upper bound: 1.4543252
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.03
Output dim: 9, lower bound: -1.4549768, upper bound: 1.4662596
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 22.03
Output dim: 9, lower bound: -1.4678653, upper bound: 1.4533667
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 22.03
Output dim: 9, lower bound: -1.4559335, upper bound: 1.4653010

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6817160, 2.6862159
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6408153, 2.6485314
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6677237, 2.6804419
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.8035312, 2.7943668
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5934188, 1.6069503
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6921091, 2.6847496
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9681692, 2.9730172
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2601280, 2.2559941
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8176346, 1.8103132
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3255701, 2.3217008

Time for backsubstitution: 12.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526397, upper bound: 1.4551966
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662382, upper bound: 1.4542393
time: 4.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6832342, 2.6846991
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6461864, 2.6431608
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6720581, 2.6761045
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.8005557, 2.7973380
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6023033, 1.5980666
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6936741, 2.6831846
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9711313, 2.9700546
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2548800, 2.2612352
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8146772, 1.8132706
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3213520, 2.3259187

Time for backsubstitution: 12.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526394, upper bound: 1.4552042
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4535980, upper bound: 1.4542475
time: 4.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6644616, 2.6589527
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6679115, 2.6642919
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6433134, 2.6521592
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7815118, 2.7774701
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6386867, 1.6337144
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6753173, 2.6686625
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9730921, 2.9652524
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2568011, 2.2609415
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8290558, 1.8243186
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3405616, 2.3450646

Time for backsubstitution: 12.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526382, upper bound: 1.4552087
time: 4.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526327, upper bound: 1.4678463
time: 5.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6568999, 2.6665144
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6642103, 2.6679931
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6458569, 2.6496153
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7816978, 2.7772837
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6337676, 1.6386328
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6782489, 2.6657281
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9649572, 2.9733882
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2566848, 2.2610571
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8274221, 1.8259518
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3424141, 2.3432124

Time for backsubstitution: 13.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4533609, upper bound: 1.4667237
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4543209, upper bound: 1.4669012
time: 4.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6781497, 2.6740913
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6738563, 2.6730657
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6535883, 2.6491685
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7777839, 2.7820063
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6463661, 1.6454384
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6727920, 2.6829100
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9857597, 2.9830041
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2611036, 2.2565532
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8284616, 1.8311238
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3468459, 2.3450651

Time for backsubstitution: 12.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4668897, upper bound: 1.4535879
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4542496, upper bound: 1.4535956
time: 4.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6786141, 2.6736264
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6749883, 2.6719337
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6546278, 2.6481285
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7768025, 2.7829866
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6487169, 1.6430881
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6731238, 2.6825781
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9865551, 2.9822087
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2595329, 2.2581244
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8278313, 1.8317540
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3456948, 2.3462162

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4549683, upper bound: 1.4662539
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4549710, upper bound: 1.4650115
time: 4.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6761441, 2.6760964
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6731477, 2.6737747
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6554708, 2.6472864
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7777791, 2.7820110
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6453848, 1.6464202
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6733203, 2.6823816
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9833002, 2.9854641
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2608099, 2.2568474
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8280201, 1.8315647
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3477161, 2.3441951

Time for backsubstitution: 12.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4667225, upper bound: 1.4533632
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4678609, upper bound: 1.4533596
time: 5.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6766086, 2.6756320
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6742797, 2.6726427
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6565104, 2.6462469
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7767987, 2.7829919
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6477356, 1.6440694
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6736522, 2.6820493
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9840946, 2.9846687
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2592382, 2.2584186
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8273907, 1.8321948
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3465650, 2.3453460

Time for backsubstitution: 12.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4552044, upper bound: 1.4526407
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4551965, upper bound: 1.4652795
time: 4.75 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 22.13 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 9, lower bound: -1.4526397, upper bound: 1.4551966
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 9, lower bound: -1.4662382, upper bound: 1.4542393
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 9, lower bound: -1.4526394, upper bound: 1.4552042
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 9, lower bound: -1.4535980, upper bound: 1.4542475
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 9, lower bound: -1.4526382, upper bound: 1.4552087
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 9, lower bound: -1.4526327, upper bound: 1.4678463
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 9, lower bound: -1.4533609, upper bound: 1.4667237
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 9, lower bound: -1.4543209, upper bound: 1.4669012
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 9, lower bound: -1.4668897, upper bound: 1.4535879
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 9, lower bound: -1.4542496, upper bound: 1.4535956
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 9, lower bound: -1.4549683, upper bound: 1.4662539
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 9, lower bound: -1.4549710, upper bound: 1.4650115
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 9, lower bound: -1.4667225, upper bound: 1.4533632
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 9, lower bound: -1.4678609, upper bound: 1.4533596
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 9, lower bound: -1.4552044, upper bound: 1.4526407
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.13
Output dim: 9, lower bound: -1.4551965, upper bound: 1.4652795

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6806798, 2.6831727
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6404295, 2.6474366
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6648788, 2.6794786
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.8035283, 2.7943592
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5929151, 1.6054652
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6912975, 2.6844659
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9668941, 2.9692826
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2599716, 2.2555425
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8174019, 1.8096399
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3242493, 2.3212504

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4652710, upper bound: 1.4551920
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4652738, upper bound: 1.4540519
time: 4.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6786733, 2.6851783
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6397209, 2.6481457
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6667604, 2.6775970
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.8035235, 2.7943649
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5919337, 1.6064465
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6918259, 2.6839375
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9644337, 2.9717426
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2596769, 2.2558367
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8169613, 1.8100808
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3251195, 2.3203802

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4649901, upper bound: 1.4542335
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4662338, upper bound: 1.4542308
time: 5.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6821961, 2.6816559
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6457996, 2.6420660
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6692123, 2.6751423
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.8005538, 2.7973304
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6017995, 1.5965812
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6928616, 2.6829009
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9698563, 2.9663205
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2547226, 2.2607837
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8144445, 1.8125973
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3200316, 2.3254681

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526309, upper bound: 1.4551998
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526337, upper bound: 1.4540600
time: 4.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6801906, 2.6836615
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6450911, 2.6427755
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6710949, 2.6732602
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.8005481, 2.7973356
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6008182, 1.5975628
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6933899, 2.6823726
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9673967, 2.9687805
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2544289, 2.2610779
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8140039, 1.8130383
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3209019, 2.3245981

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4523493, upper bound: 1.4542414
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4535937, upper bound: 1.4542388
time: 4.70 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6695089, 2.6655178
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6356983, 2.6374497
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6619453, 2.6751256
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.8020434, 2.7950315
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5875330, 1.5914450
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6845646, 2.6794758
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9553185, 2.9504399
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2583466, 2.2572463
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8142624, 1.8065679
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3194647, 2.3197496

Time for backsubstitution: 12.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526288, upper bound: 1.4552018
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4523472, upper bound: 1.4542461
time: 5.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6710262, 2.6640005
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6410694, 2.6320786
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6662817, 2.6707911
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7990727, 2.7980061
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5964170, 1.5825605
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6861305, 2.6779108
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9582806, 2.9474778
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2531061, 2.2624946
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8113050, 1.8095253
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3152466, 2.3239675

Time for backsubstitution: 12.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526209, upper bound: 1.4678405
time: 5.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4523417, upper bound: 1.4668843
time: 5.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6558628, 2.6634707
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6638250, 2.6669002
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6430120, 2.6486521
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7816834, 2.7772751
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6332641, 1.6371477
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6774373, 2.6654444
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9636822, 2.9696541
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2565284, 2.2606058
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8271894, 1.8252785
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3410919, 2.3427608

Time for backsubstitution: 12.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526316, upper bound: 1.4540634
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4526260, upper bound: 1.4667026
time: 5.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6538563, 2.6654763
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6631165, 2.6676188
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6448936, 2.6467700
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7816892, 2.7772803
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6322827, 1.6381292
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6779723, 2.6649160
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9612226, 2.9721141
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2562337, 2.2609034
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8267488, 1.8257220
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3419626, 2.3418906

Time for backsubstitution: 12.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4535916, upper bound: 1.4542409
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4535861, upper bound: 1.4668823
time: 5.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6831965, 2.6806555
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6416435, 2.6462231
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6722202, 2.6721373
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7983203, 2.7995677
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5952125, 1.6031680
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6820412, 2.6937227
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9679852, 2.9681916
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2626562, 2.2528572
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8136683, 1.8133733
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3257494, 2.3197508

Time for backsubstitution: 12.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4668812, upper bound: 1.4535837
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4668839, upper bound: 1.4523390
time: 4.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6847138, 2.6791382
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6470137, 2.6408529
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6765547, 2.6678004
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7953458, 2.8025389
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6040969, 1.5942843
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6836052, 2.6921582
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9709473, 2.9652290
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2574081, 2.2580984
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8107109, 1.8163307
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3215313, 2.3239686

Time for backsubstitution: 12.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4542411, upper bound: 1.4535915
time: 4.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4542437, upper bound: 1.4523470
time: 5.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6659412, 2.6533923
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6687508, 2.6619849
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6478090, 2.6438541
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7762990, 2.7826695
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6404796, 1.6299322
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6652474, 2.6776400
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9729090, 2.9604273
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2593322, 2.2578049
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8250914, 1.8273790
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3407395, 2.3431134

Time for backsubstitution: 12.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4542390, upper bound: 1.4535936
time: 5.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4542311, upper bound: 1.4662337
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6583786, 2.6609530
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6650391, 2.6656866
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6503534, 2.6413102
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7764754, 2.7824836
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6355615, 1.6348512
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6681819, 2.6747012
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9647741, 2.9685631
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2592130, 2.2579205
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8234558, 1.8290126
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3425920, 2.3412611

Time for backsubstitution: 12.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4542417, upper bound: 1.4523505
time: 4.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4542338, upper bound: 1.4649900
time: 4.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6634712, 2.6558623
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6668997, 2.6638255
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6486521, 2.6430116
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7772756, 2.7816834
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6371474, 1.6332643
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6654439, 2.6774373
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9696541, 2.9636822
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2606063, 2.2565279
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8252783, 1.8271894
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3427608, 2.3410921

Time for backsubstitution: 12.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4667023, upper bound: 1.4526233
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4540622, upper bound: 1.4526316
time: 5.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6559086, 2.6634231
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6631985, 2.6675372
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6511955, 2.6404676
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7774615, 2.7815080
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6322293, 1.6381834
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6683850, 2.6745048
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9615183, 2.9718180
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2604899, 2.2566471
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8236456, 1.8288257
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3446128, 2.3392398

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4540625, upper bound: 1.4526213
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4552020, upper bound: 1.4526287
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6816564, 2.6821961
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6420660, 2.6458001
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6751423, 2.6692128
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7973304, 2.8005538
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5965810, 1.6017995
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6829014, 2.6928620
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9663210, 2.9698563
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2607841, 2.2547226
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8125973, 1.8144443
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3254681, 2.3200316

Time for backsubstitution: 12.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4523496, upper bound: 1.4526329
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4552000, upper bound: 1.4526322
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6831727, 2.6806793
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6474361, 2.6404295
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6794786, 2.6648788
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7943597, 2.8035288
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.6054654, 1.5929151
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6844654, 2.6912975
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9692831, 2.9668941
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2555428, 2.2599709
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8096399, 1.8174019
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3212504, 2.3242495

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4540522, upper bound: 1.4652735
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4540546, upper bound: 1.4652714
time: 5.95 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.74 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4652710, upper bound: 1.4551920
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4652738, upper bound: 1.4540519
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4649901, upper bound: 1.4542335
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4662338, upper bound: 1.4542308
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4526309, upper bound: 1.4551998
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4526337, upper bound: 1.4540600
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4523493, upper bound: 1.4542414
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4535937, upper bound: 1.4542388
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4526288, upper bound: 1.4552018
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4523472, upper bound: 1.4542461
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4526209, upper bound: 1.4678405
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4523417, upper bound: 1.4668843
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4526316, upper bound: 1.4540634
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4526260, upper bound: 1.4667026
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4535916, upper bound: 1.4542409
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4535861, upper bound: 1.4668823
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4668812, upper bound: 1.4535837
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4668839, upper bound: 1.4523390
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4542411, upper bound: 1.4535915
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4542437, upper bound: 1.4523470
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4542390, upper bound: 1.4535936
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4542311, upper bound: 1.4662337
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4542417, upper bound: 1.4523505
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4542338, upper bound: 1.4649900
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4667023, upper bound: 1.4526233
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4540622, upper bound: 1.4526316
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4540625, upper bound: 1.4526213
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4552020, upper bound: 1.4526287
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4523496, upper bound: 1.4526329
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4552000, upper bound: 1.4526322
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4540522, upper bound: 1.4652735
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.74
Output dim: 9, lower bound: -1.4540546, upper bound: 1.4652714

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6680069, 2.6629386
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6341925, 2.6374874
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6580601, 2.6752038
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.8030257, 2.7940431
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5846791, 1.5923097
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6834221, 2.6795306
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9532466, 2.9475002
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2597709, 2.2552235
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8146625, 1.8052640
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3192945, 2.3181477

Time for backsubstitution: 12.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 1845

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1978

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4593373, upper bound: 1.4539406
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4640346, upper bound: 1.4494361
time: 6.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6604443, 2.6704998
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6304808, 2.6411891
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6606035, 2.6726604
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.8032022, 2.7938566
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5797601, 1.5972283
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6863537, 2.6765900
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9451118, 2.9556360
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2596517, 2.2553394
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8130260, 1.8068969
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3211470, 2.3162954

Time for backsubstitution: 12.79 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2235

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2615

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4487945, upper bound: 1.4383795
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4495131, upper bound: 1.4376678
time: 6.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.6660004, 2.6649442
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.6334734, 2.6381965
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.6599417, 2.6733217
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.8030210, 2.7940378
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5836978, 1.5932913
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.6839504, 2.6789961
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.9507880, 2.9499602
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.2594733, 2.2555177
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.8142190, 1.8057048
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.3201647, 2.3172774

Time for backsubstitution: 12.71 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.3557186126708984
rel_dist={9: [-1.4681627927167926, 1.4681601773366633]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 833
type: RSZ, layer: 1, pos: 6222
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 833

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597462, upper bound: 1.3509753
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3509758, upper bound: 1.3597455
time: 7.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.31 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.31
Output dim: 9, lower bound: -1.3597462, upper bound: 1.3509753
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.31
Output dim: 9, lower bound: -1.3509758, upper bound: 1.3597455

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5327606, 2.5331092
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5562525, 2.5571017
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5457487, 2.5465288
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7194481, 2.7187123
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5780787, 1.5798419
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5627742, 2.5630231
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8603477, 2.8609447
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1834188, 2.1822405
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7701550, 1.7696826
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2939169, 2.2930536

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 6222
type: RSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597312, upper bound: 1.3502367
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3502451, upper bound: 1.3502429
time: 5.96 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5331087, 2.5327611
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5571012, 2.5562525
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5465288, 2.5457487
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7187128, 2.7194481
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5798421, 1.5780790
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5630221, 2.5627737
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8609447, 2.8603482
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1822405, 2.1834188
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7696829, 1.7701552
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2930539, 2.2939169

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5747
type: RSZ, layer: 1, pos: 6222
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5747

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3502434, upper bound: 1.3502441
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3502399, upper bound: 1.3597312
time: 8.95 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 29.26 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.26
Output dim: 9, lower bound: -1.3597312, upper bound: 1.3502367
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 29.26
Output dim: 9, lower bound: -1.3502451, upper bound: 1.3502429
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 29.26
Output dim: 9, lower bound: -1.3502434, upper bound: 1.3502441
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.26
Output dim: 9, lower bound: -1.3502399, upper bound: 1.3597312

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5378084, 2.5392952
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5240407, 2.5289168
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5643806, 2.5684128
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7392397, 2.7362733
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5269258, 1.5353513
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5720234, 2.5734458
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8425732, 2.8453908
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1836605, 2.1785457
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7546220, 1.7519314
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2717667, 2.2677398

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3592411, upper bound: 1.3502323
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597273, upper bound: 1.3497485
time: 4.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5392952, 2.5378084
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5289168, 2.5240402
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5684128, 2.5643802
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7362728, 2.7392397
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5353515, 1.5269256
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5734453, 2.5720234
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8453913, 2.8425722
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1785460, 2.1836603
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7519317, 1.7546222
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2677398, 2.2717664

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3497504, upper bound: 1.3597273
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3502336, upper bound: 1.3592417
time: 4.78 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 9, lower bound: -1.3592411, upper bound: 1.3502323
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 9, lower bound: -1.3597273, upper bound: 1.3497485
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 9, lower bound: -1.3497504, upper bound: 1.3597273
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.97
Output dim: 9, lower bound: -1.3502336, upper bound: 1.3592417

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5232468, 2.5190611
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5168667, 2.5189667
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5575619, 2.5635018
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7387371, 2.7359104
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5174594, 1.5221958
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5641470, 2.5677700
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8268929, 2.8236089
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1834278, 2.1782265
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7514710, 1.7475555
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2668123, 2.2641754

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3592357, upper bound: 1.3502290
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3590531, upper bound: 1.3497440
time: 6.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5175743, 2.5247321
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5140896, 2.5217428
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5594692, 2.5615940
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7388773, 2.7357707
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5137696, 1.5258851
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5663471, 2.5655699
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8207903, 2.8297110
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1833410, 2.1783133
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7502465, 1.7487803
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2682018, 2.2627861

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3583197, upper bound: 1.3497498
time: 8.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3483487, upper bound: 1.3483395
time: 8.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5377555, 2.5347652
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5283546, 2.5229459
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5655689, 2.5629478
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7362700, 2.7392321
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5346022, 1.5254400
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5726337, 2.5716076
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8435011, 2.8388381
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1783152, 2.1832087
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7515888, 1.7539489
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2664175, 2.2710969

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3497435, upper bound: 1.3597230
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3497447, upper bound: 1.3590531
time: 5.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5362525, 2.5362697
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5278225, 2.5234771
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5669804, 2.5615358
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7362652, 2.7392364
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5338659, 1.5261762
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5730305, 2.5712113
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8416567, 2.8406830
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1780939, 2.1834292
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7512579, 1.7542796
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2670698, 2.2704444

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5816
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5816

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3495630, upper bound: 1.3592358
time: 5.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3502297, upper bound: 1.3592348
time: 5.71 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.93
Output dim: 9, lower bound: -1.3592357, upper bound: 1.3502290
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.93
Output dim: 9, lower bound: -1.3590531, upper bound: 1.3497440
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.93
Output dim: 9, lower bound: -1.3583197, upper bound: 1.3497498
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 25.93
Output dim: 9, lower bound: -1.3483487, upper bound: 1.3483395
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.93
Output dim: 9, lower bound: -1.3497435, upper bound: 1.3597230
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.93
Output dim: 9, lower bound: -1.3497447, upper bound: 1.3590531
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.93
Output dim: 9, lower bound: -1.3495630, upper bound: 1.3592358
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.93
Output dim: 9, lower bound: -1.3502297, upper bound: 1.3592348

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5217056, 2.5160174
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5163116, 2.5178733
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5547161, 2.5620685
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7387333, 2.7359028
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5167103, 1.5207107
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5633354, 2.5673599
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8250022, 2.8198738
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1831994, 2.1777749
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7511306, 1.7468827
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2654910, 2.2635064

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3578287, upper bound: 1.3502279
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3592341, upper bound: 1.3488234
time: 4.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5202026, 2.5175214
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5157728, 2.5184050
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5561275, 2.5606565
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7387295, 2.7358990
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5159740, 1.5214469
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5637321, 2.5669589
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8231568, 2.8217187
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1829762, 2.1779954
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7507977, 1.7472131
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2661438, 2.2628539

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3480506, upper bound: 1.3497421
time: 7.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3590506, upper bound: 1.3483358
time: 5.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5120440, 2.5210896
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5114155, 2.5199790
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5433435, 2.5509748
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7313304, 2.7243180
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5087271, 1.5225644
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5529327, 2.5452132
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8183918, 2.8281312
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1774292, 2.1744151
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7448263, 1.7405598
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2649021, 2.2606115

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 902

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 902

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3578300, upper bound: 1.3495571
time: 8.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3583159, upper bound: 1.3497435
time: 5.38 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5231924, 2.5145311
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5211887, 2.5129967
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5587482, 2.5580359
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7357674, 2.7388697
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5251360, 1.5122850
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5647583, 2.5659370
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8278203, 2.8170552
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1780849, 2.1828895
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7484393, 1.7495735
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2614646, 2.2675331

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3483349, upper bound: 1.3597212
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3497420, upper bound: 1.3583158
time: 4.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5175209, 2.5202022
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5184050, 2.5157728
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5606565, 2.5561280
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7358990, 2.7387295
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5214467, 1.5159743
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5669584, 2.5637317
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8217187, 2.8231573
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1779952, 2.1829762
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7472129, 1.7507980
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2628541, 2.2661438

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3483362, upper bound: 1.3590500
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3497432, upper bound: 1.3575358
time: 5.34 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5216885, 2.5160351
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5206499, 2.5135279
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5601597, 2.5566239
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7357635, 2.7388659
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5244002, 1.5130210
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5651541, 2.5655365
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8259759, 2.8189001
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1778617, 2.1831100
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7481074, 1.7499039
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2621169, 2.2668803

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3480446, upper bound: 1.3592343
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3480446, upper bound: 1.3578297
time: 5.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5160179, 2.5217061
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5178728, 2.5163116
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5620680, 2.5547166
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7359028, 2.7387338
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5207105, 1.5167103
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5673599, 2.5633354
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8198733, 2.8250022
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1777749, 2.1831994
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7468829, 1.7511306
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2635064, 2.2654912

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6222

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6222

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3488210, upper bound: 1.3592336
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3502282, upper bound: 1.3578279
time: 4.77 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 24.40 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.40
Output dim: 9, lower bound: -1.3578287, upper bound: 1.3502279
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.40
Output dim: 9, lower bound: -1.3592341, upper bound: 1.3488234
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 24.40
Output dim: 9, lower bound: -1.3480506, upper bound: 1.3497421
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.40
Output dim: 9, lower bound: -1.3590506, upper bound: 1.3483358
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.40
Output dim: 9, lower bound: -1.3578300, upper bound: 1.3495571
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.40
Output dim: 9, lower bound: -1.3583159, upper bound: 1.3497435
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.40
Output dim: 9, lower bound: -1.3483349, upper bound: 1.3597212
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.40
Output dim: 9, lower bound: -1.3497420, upper bound: 1.3583158
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.40
Output dim: 9, lower bound: -1.3483362, upper bound: 1.3590500
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.40
Output dim: 9, lower bound: -1.3497432, upper bound: 1.3575358
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.40
Output dim: 9, lower bound: -1.3480446, upper bound: 1.3592343
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.40
Output dim: 9, lower bound: -1.3480446, upper bound: 1.3578297
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.40
Output dim: 9, lower bound: -1.3488210, upper bound: 1.3592336
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.40
Output dim: 9, lower bound: -1.3502282, upper bound: 1.3578279

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5161762, 2.5123749
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5136385, 2.5161090
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5385904, 2.5514483
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7311873, 2.7244501
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5116677, 1.5173905
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5499210, 2.5470033
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8226047, 2.8182940
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1772876, 2.1738768
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7457099, 1.7386613
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2621922, 2.2613320

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2319

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3529440, upper bound: 1.3495582
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3571569, upper bound: 1.3453394
time: 5.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5180645, 2.5104871
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5145483, 2.5151987
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5440960, 2.5459418
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7272811, 2.7283564
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5133901, 1.5156677
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5429783, 2.5539446
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8234229, 2.8174753
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1793017, 2.1718628
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7429090, 1.7414615
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2633171, 2.2602074

Time for backsubstitution: 14.95 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1978

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3205864, upper bound: 1.3109679
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3205864, upper bound: 1.3109679
time: 4.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5165596, 2.5119910
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5140085, 2.5157304
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5455074, 2.5445304
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7272773, 2.7283525
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5126538, 1.5164039
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5433750, 2.5535436
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8215785, 2.8193202
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1790786, 2.1720834
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7425771, 1.7417920
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2639694, 2.2595549

Time for backsubstitution: 14.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 920

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1843

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3302558, upper bound: 1.3167295
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3273321, upper bound: 1.3195974
time: 5.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5105057, 2.5180459
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5108538, 2.5188851
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5404978, 2.5495405
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7313190, 2.7243104
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5079780, 1.5210793
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5521212, 2.5447979
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8165030, 2.8243957
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1771979, 2.1739635
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7444825, 1.7398860
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2635813, 2.2599430

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1843

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2811

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3465346, upper bound: 1.3431917
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3514792, upper bound: 1.3381722
time: 5.18 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5090008, 2.5195498
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5103216, 2.5194240
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5419092, 2.5481291
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7313228, 2.7243147
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5072417, 1.5218155
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5525217, 2.5444012
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8146567, 2.8262405
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1769776, 2.1741867
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7441525, 1.7402184
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2642341, 2.2592902

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 313

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 920

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3571004, upper bound: 1.3484731
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3571474, upper bound: 1.3480075
time: 4.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5176620, 2.5108886
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5185137, 2.5112324
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5426226, 2.5474157
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7282205, 2.7274170
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5200934, 1.5089648
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5513439, 2.5455809
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8254218, 2.8154755
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1721730, 2.1789913
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7430186, 1.7413521
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2581654, 2.2653587

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1208

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3475123, upper bound: 1.3536329
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3422425, upper bound: 1.3589054
time: 5.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5195503, 2.5090003
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5194254, 2.5103221
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5481291, 2.5419097
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7243142, 2.7313232
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5218158, 1.5072420
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5444012, 2.5525222
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8262401, 2.8146567
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1741872, 2.1769774
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7402186, 1.7441523
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2592902, 2.2642341

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 2328

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 975

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3487090, upper bound: 1.3583119
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3497386, upper bound: 1.3572839
time: 5.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5119915, 2.5165596
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5157309, 2.5140085
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5445309, 2.5455079
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7283521, 2.7272768
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5164037, 1.5126536
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5535431, 2.5433750
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8193202, 2.8215775
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1720834, 2.1790781
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7417922, 1.7425768
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2595549, 2.2639694

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1101

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3274356, upper bound: 1.3382356
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3274429, upper bound: 1.3382379
time: 6.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5138798, 2.5146718
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5166407, 2.5130982
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5500364, 2.5400019
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7244458, 2.7311831
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5181270, 1.5109313
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5466022, 2.5503178
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8201385, 2.8207593
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1740975, 2.1770642
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7389922, 1.7453775
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2606792, 2.2628448

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 337

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 313

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3161311, upper bound: 1.3239130
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3161311, upper bound: 1.3239130
time: 6.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5161572, 2.5123925
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5179758, 2.5117640
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5440340, 2.5460043
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7282166, 2.7274132
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5193572, 1.5097008
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5517406, 2.5451794
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8235774, 2.8173203
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1719499, 2.1792119
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7426867, 1.7416828
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2588181, 2.2647061

Time for backsubstitution: 14.31 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 1101

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2578

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3363288, upper bound: 1.3469920
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3361312, upper bound: 1.3471686
time: 5.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5180454, 2.5105047
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5188856, 2.5108538
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5495405, 2.5404983
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7243104, 2.7313194
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5210795, 1.5079782
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5447979, 2.5521212
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8243957, 2.8165021
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1739640, 2.1771979
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7398858, 1.7444828
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2599430, 2.2635813

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 1844

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2908

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3492614, upper bound: 1.3574407
time: 7.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3490512, upper bound: 1.3575095
time: 7.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5104866, 2.5180635
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5151987, 2.5145478
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5459423, 2.5440965
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7283559, 2.7272811
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5156674, 1.5133898
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5539446, 2.5429788
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8174758, 2.8234224
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1718631, 2.1793013
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7414613, 1.7429092
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2602072, 2.2633169

Time for backsubstitution: 14.73 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 3109
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 2536

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1844

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3450732, upper bound: 1.3527833
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3440976, upper bound: 1.3540764
time: 5.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5123749, 2.5161757
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5161085, 2.5136371
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5514479, 2.5385900
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7244496, 2.7311873
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5173907, 1.5116675
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5470028, 2.5499215
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8182940, 2.8226037
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1738772, 2.1772873
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7386613, 1.7457099
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2613320, 2.2621922

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 906
type: RSZ, layer: 3, pos: 1384
type: RSZ, layer: 3, pos: 208
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1690
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 1445
type: RSZ, layer: 3, pos: 2235
type: RSZ, layer: 3, pos: 2222
type: RSZ, layer: 3, pos: 1208
type: RSZ, layer: 3, pos: 2526
type: RSZ, layer: 3, pos: 2123
type: RSZ, layer: 3, pos: 2126
type: RSZ, layer: 3, pos: 2536
type: RSZ, layer: 3, pos: 2811
type: RSZ, layer: 3, pos: 2319
type: RSZ, layer: 3, pos: 920
type: RSZ, layer: 3, pos: 1843
type: RSZ, layer: 3, pos: 1851
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 2628
type: RSZ, layer: 3, pos: 1095
type: RSZ, layer: 3, pos: 2328
type: RSZ, layer: 3, pos: 2578
type: RSZ, layer: 3, pos: 1983
type: RSZ, layer: 3, pos: 1845
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 564
type: RSZ, layer: 3, pos: 1244
type: RSZ, layer: 3, pos: 1101
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 2908
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 1844
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 1739
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2615
type: RSZ, layer: 3, pos: 909
type: RSZ, layer: 3, pos: 2004
type: RSZ, layer: 3, pos: 947
type: RSZ, layer: 3, pos: 2223
type: RSZ, layer: 3, pos: 160
type: RSZ, layer: 3, pos: 2516
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 3109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 906

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.2787816, upper bound: 1.2869856
time: 5.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.2787816, upper bound: 1.2869856
time: 4.98 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 24.51 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3529440, upper bound: 1.3495582
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3571569, upper bound: 1.3453394
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3205864, upper bound: 1.3109679
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3205864, upper bound: 1.3109679
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3302558, upper bound: 1.3167295
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3273321, upper bound: 1.3195974
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3465346, upper bound: 1.3431917
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3514792, upper bound: 1.3381722
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3571004, upper bound: 1.3484731
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3571474, upper bound: 1.3480075
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3475123, upper bound: 1.3536329
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3422425, upper bound: 1.3589054
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3487090, upper bound: 1.3583119
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3497386, upper bound: 1.3572839
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3274356, upper bound: 1.3382356
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3274429, upper bound: 1.3382379
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3161311, upper bound: 1.3239130
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3161311, upper bound: 1.3239130
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3363288, upper bound: 1.3469920
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3361312, upper bound: 1.3471686
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3492614, upper bound: 1.3574407
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3490512, upper bound: 1.3575095
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3450732, upper bound: 1.3527833
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.3440976, upper bound: 1.3540764
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.2787816, upper bound: 1.2869856
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 24.51
Output dim: 9, lower bound: -1.2787816, upper bound: 1.2869856

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.4948936, 2.4975171
1: -12.4945774, -8.9361620, -12.4945774, -8.9361620, -2.5138206, 2.5142684
2: -13.4097614, -10.1796103, -13.4097614, -10.1796103, -2.5143304, 2.5282536
3: -9.8902388, -6.9025407, -9.8902388, -6.9025407, -2.7210441, 2.7174339
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5072751, 1.5116222
5: -11.0733919, -7.3661022, -11.0733919, -7.3661022, -2.5269766, 2.5209398
6: -17.5802174, -13.6031437, -17.5802174, -13.6031437, -2.8146081, 2.8098640
7: -6.4332151, -3.5954418, -6.4332151, -3.5954418, -2.1695070, 2.1634772
8: -2.0399036, 0.1837792, -2.0399036, 0.1837792, -1.7366514, 1.7322395
9: 2.4171557, 5.1602306, 2.4171557, 5.1602306, -2.2569413, 2.2538285

Time for backsubstitution: 14.65 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.2999587059020996
rel_dist={9: [-1.3600305867793656, 1.360028066183916]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2417.22 seconds
