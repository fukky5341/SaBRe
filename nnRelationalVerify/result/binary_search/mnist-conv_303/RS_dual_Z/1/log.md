## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.38438213844
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801)
1: (-10.8713818, -7.8377485, -10.8713818, -7.8377485, -3.0336332, 3.0336332)
2: (-5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.7213984, 2.7213984)
3: (-6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040)
4: (-13.4648161, -9.8270741, -13.4648161, -9.8270741, -3.3610158, 3.3610163)
5: (-3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.8814673, 1.8814671)
6: (-10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.8215880, 2.8215880)
7: (-9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965)
8: (9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.6745625, 2.6745625)
9: (-7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.3670969, 3.3670969)

## BASE Result
execution time: IAR + LP analysis = 12.99 + 58.72 = 71.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -1.9845428, upper bound: 1.9845425


# Binary Search by BASE starts (time budget: 3528.30 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.359529972076416
rel_dist={8: [-1.3847687820365167, 1.3847690605870273]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.1324596405029297
rel_dist={8: [-1.0332529041562815, 1.0332525329195708]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=2.2081499099731445
rel_dist={8: [-1.1524706431871987, 1.1524726628995694]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=2.2838401794433594
rel_dist={8: [-1.269933294744563, 1.2699326298994738]}

## Binary Search Result
Binary search time: 194.66 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_dual_Z) starts
Time budget: 3333.64 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7090557, upper bound: 1.7090557
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7090557, upper bound: 1.7090557
time: 7.02 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.20 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.20
Output dim: 8, lower bound: -1.7090557, upper bound: 1.7090557
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.20
Output dim: 8, lower bound: -1.7090557, upper bound: 1.7090557

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8868480, 2.8842297
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5732374, 2.5715857
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9934125, 2.9989176
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6903834, 1.6943862
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5399652, 2.5475075
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5865889, 2.5866127
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1320624, 3.1281776

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4556

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7045345, upper bound: 1.7090345
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7090346, upper bound: 1.7045340
time: 6.06 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8842292, 2.8853202
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5715857, 2.5722775
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9957137, 2.9934125
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6920609, 1.6903834
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5431194, 2.5399654
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5866003, 2.5865889
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1281781, 3.1298084

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4556

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7045345, upper bound: 1.7090345
time: 6.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7090346, upper bound: 1.7045341
time: 6.15 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.88 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.88
Output dim: 8, lower bound: -1.7045345, upper bound: 1.7090345
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.88
Output dim: 8, lower bound: -1.7090346, upper bound: 1.7045340
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.88
Output dim: 8, lower bound: -1.7045345, upper bound: 1.7090345
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.88
Output dim: 8, lower bound: -1.7090346, upper bound: 1.7045341

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8930364, 2.8838582
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5505581, 2.5628867
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9855609, 2.9717026
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6738298, 1.6668317
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5287352, 2.5259860
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5791025, 2.5819368
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1186004, 3.1197596

Time for backsubstitution: 12.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7045222, upper bound: 1.7015184
time: 8.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6970199, upper bound: 1.7090215
time: 7.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8864770, 2.8904176
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5645390, 2.5489058
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9661975, 2.9910665
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6628292, 1.6778326
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5184441, 2.5362771
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5819130, 2.5791268
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1236453, 3.1147156

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7090213, upper bound: 1.6970217
time: 16.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7015185, upper bound: 1.7045217
time: 6.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8904176, 2.8849483
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5489063, 2.5635791
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9878640, 2.9661975
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6755059, 1.6628289
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5318885, 2.5184438
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5791140, 2.5819130
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1147151, 3.1213903

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7045222, upper bound: 1.7015184
time: 8.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6970199, upper bound: 1.7090215
time: 7.80 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8838582, 2.8915076
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5628872, 2.5495980
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9684997, 2.9855611
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6645052, 1.6738299
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5215974, 2.5287349
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5819235, 2.5791030
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1197600, 3.1163464

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7090213, upper bound: 1.6970202
time: 16.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7015185, upper bound: 1.7045217
time: 6.18 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 35.20 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 35.20
Output dim: 8, lower bound: -1.7045222, upper bound: 1.7015184
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 35.20
Output dim: 8, lower bound: -1.6970199, upper bound: 1.7090215
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 35.20
Output dim: 8, lower bound: -1.7090213, upper bound: 1.6970217
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 35.20
Output dim: 8, lower bound: -1.7015185, upper bound: 1.7045217
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 35.20
Output dim: 8, lower bound: -1.7045222, upper bound: 1.7015184
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 35.20
Output dim: 8, lower bound: -1.6970199, upper bound: 1.7090215
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 35.20
Output dim: 8, lower bound: -1.7090213, upper bound: 1.6970202
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 35.20
Output dim: 8, lower bound: -1.7015185, upper bound: 1.7045217

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8921590, 2.8837709
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5502110, 2.5589893
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9665971, 2.9700253
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6729074, 1.6558940
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5285139, 2.5234811
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5782466, 2.5718727
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1051078, 3.1186228

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7044990, upper bound: 1.7009400
time: 6.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7039421, upper bound: 1.7014954
time: 6.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8929505, 2.8829813
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5466604, 2.5625401
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9838872, 2.9527383
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6628923, 1.6659112
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5262299, 2.5257654
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5690389, 2.5810809
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1174655, 3.1062675

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6969967, upper bound: 1.7084411
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6964398, upper bound: 1.7089999
time: 5.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8855996, 2.8903313
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5641918, 2.5450082
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9472337, 2.9893930
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6619086, 1.6668949
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5182238, 2.5337722
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5810571, 2.5690627
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1101527, 3.1135798

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7089981, upper bound: 1.6964395
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7084412, upper bound: 1.6969966
time: 9.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8863893, 2.8895407
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5606413, 2.5485587
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9645200, 2.9721022
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6518912, 1.6769103
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5159388, 2.5360565
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5718493, 2.5782704
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1225085, 3.1012235

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7014953, upper bound: 1.7039421
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7009408, upper bound: 1.7044988
time: 6.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8895402, 2.8848610
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5485592, 2.5596812
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9688993, 2.9645200
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6745844, 1.6518912
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5316677, 2.5159390
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5782571, 2.5718489
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1012225, 3.1202536

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7044990, upper bound: 1.7009405
time: 6.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7039421, upper bound: 1.7014954
time: 6.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8903317, 2.8840714
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5450077, 2.5632319
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9861903, 2.9472332
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6645689, 1.6619084
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5293837, 2.5182233
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5690494, 2.5810571
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1135802, 3.1078982

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6969967, upper bound: 1.7084411
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6964398, upper bound: 1.7089999
time: 5.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8829808, 2.8914213
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5625401, 2.5457001
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9495358, 2.9838877
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6635847, 1.6628922
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5213771, 2.5262301
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5810676, 2.5690389
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1062675, 3.1152105

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7089981, upper bound: 1.6964395
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7084412, upper bound: 1.6969966
time: 7.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8837705, 2.8906307
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5589895, 2.5492506
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9668231, 2.9665968
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6535673, 1.6729076
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5190926, 2.5285141
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5718589, 2.5782466
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1186233, 3.1028543

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7014953, upper bound: 1.7039419
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7009408, upper bound: 1.7044989
time: 6.44 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.77
Output dim: 8, lower bound: -1.7044990, upper bound: 1.7009400
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.77
Output dim: 8, lower bound: -1.7039421, upper bound: 1.7014954
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.77
Output dim: 8, lower bound: -1.6969967, upper bound: 1.7084411
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.77
Output dim: 8, lower bound: -1.6964398, upper bound: 1.7089999
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.77
Output dim: 8, lower bound: -1.7089981, upper bound: 1.6964395
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.77
Output dim: 8, lower bound: -1.7084412, upper bound: 1.6969966
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.77
Output dim: 8, lower bound: -1.7014953, upper bound: 1.7039421
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.77
Output dim: 8, lower bound: -1.7009408, upper bound: 1.7044988
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.77
Output dim: 8, lower bound: -1.7044990, upper bound: 1.7009405
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.77
Output dim: 8, lower bound: -1.7039421, upper bound: 1.7014954
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.77
Output dim: 8, lower bound: -1.6969967, upper bound: 1.7084411
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.77
Output dim: 8, lower bound: -1.6964398, upper bound: 1.7089999
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.77
Output dim: 8, lower bound: -1.7089981, upper bound: 1.6964395
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.77
Output dim: 8, lower bound: -1.7084412, upper bound: 1.6969966
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.77
Output dim: 8, lower bound: -1.7014953, upper bound: 1.7039419
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.77
Output dim: 8, lower bound: -1.7009408, upper bound: 1.7044989

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8946085, 2.8827252
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5471783, 2.5660973
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9773278, 2.9654295
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6799431, 1.6528947
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5282364, 2.5241282
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5771003, 2.5745945
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1154656, 3.1142025

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6982080, upper bound: 1.7009304
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7044887, upper bound: 1.6946513
time: 10.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8911142, 2.8837709
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5502110, 2.5559564
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9620013, 2.9700253
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6699085, 1.6558940
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5285139, 2.5232031
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5782466, 2.5707254
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1006885, 3.1186228

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6976511, upper bound: 1.7014853
time: 6.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7039318, upper bound: 1.6952050
time: 7.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8954000, 2.8819356
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5436268, 2.5696478
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9946189, 2.9481425
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6699276, 1.6629119
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5259523, 2.5264122
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5678916, 2.5838027
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1278224, 3.1018476

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6907058, upper bound: 1.7084308
time: 13.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6969865, upper bound: 1.7021511
time: 6.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8919039, 2.8829813
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5466604, 2.5595069
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9792914, 2.9527383
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6598930, 1.6659112
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5262299, 2.5254874
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5690389, 2.5799341
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1130452, 3.1062675

Time for backsubstitution: 12.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6901489, upper bound: 1.7089883
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6964296, upper bound: 1.7027078
time: 7.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8880491, 2.8892856
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5611591, 2.5521162
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9579644, 2.9847972
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6689434, 1.6638957
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5179453, 2.5344191
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5799098, 2.5717845
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1205096, 3.1091599

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7027080, upper bound: 1.6964297
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7089882, upper bound: 1.6901489
time: 7.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8845549, 2.8903313
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5641918, 2.5419753
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9426379, 2.9893930
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6589093, 1.6668949
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5182238, 2.5334942
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5810571, 2.5679159
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1057324, 3.1135798

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7021511, upper bound: 1.6969864
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7084312, upper bound: 1.6907060
time: 9.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8888388, 2.8884950
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5576086, 2.5556669
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9752517, 2.9675064
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6589265, 1.6739111
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5156612, 2.5367033
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5707021, 2.5809922
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1328654, 3.0968037

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6952049, upper bound: 1.7039314
time: 7.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7014853, upper bound: 1.6976511
time: 7.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8853445, 2.8895407
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5606413, 2.5455260
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9599242, 2.9721022
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6488919, 1.6769103
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5159388, 2.5357785
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5718493, 2.5771236
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1180882, 3.1012235

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6946495, upper bound: 1.7044889
time: 8.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7009307, upper bound: 1.6982074
time: 7.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8919897, 2.8838167
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5455265, 2.5667889
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9796309, 2.9599242
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6816192, 1.6488920
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5313902, 2.5165858
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5771117, 2.5745707
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1115804, 3.1158333

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6982080, upper bound: 1.7009304
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7044887, upper bound: 1.6946513
time: 10.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8884954, 2.8848610
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5485592, 2.5566483
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9643044, 2.9645200
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6715846, 1.6518912
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5316677, 2.5156610
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5782571, 2.5707021
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.0968032, 3.1202536

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6976511, upper bound: 1.7014853
time: 6.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7039318, upper bound: 1.6952050
time: 7.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8927813, 2.8830271
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5419750, 2.5703397
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9969220, 2.9426374
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6716037, 1.6589092
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5291057, 2.5188701
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5679030, 2.5837789
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1239371, 3.1034784

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6907058, upper bound: 1.7084310
time: 9.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6969865, upper bound: 1.7021511
time: 6.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8892851, 2.8840714
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5450077, 2.5601988
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9815946, 2.9472332
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6615691, 1.6619084
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5293837, 2.5179453
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5690494, 2.5799103
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1091599, 3.1078982

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6901489, upper bound: 1.7089883
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6964296, upper bound: 1.7027078
time: 7.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8854303, 2.8903766
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5595074, 2.5528078
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9602666, 2.9792919
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6706195, 1.6598930
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5210991, 2.5268769
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5799212, 2.5717607
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1166253, 3.1107907

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7027080, upper bound: 1.6964297
time: 6.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7089882, upper bound: 1.6901489
time: 7.42 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 26.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.6982080, upper bound: 1.7009304
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.7044887, upper bound: 1.6946513
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.6976511, upper bound: 1.7014853
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.7039318, upper bound: 1.6952050
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.6907058, upper bound: 1.7084308
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.6969865, upper bound: 1.7021511
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.6901489, upper bound: 1.7089883
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.6964296, upper bound: 1.7027078
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.7027080, upper bound: 1.6964297
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.7089882, upper bound: 1.6901489
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.7021511, upper bound: 1.6969864
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.7084312, upper bound: 1.6907060
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.6952049, upper bound: 1.7039314
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.7014853, upper bound: 1.6976511
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.6946495, upper bound: 1.7044889
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.7009307, upper bound: 1.6982074
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.6982080, upper bound: 1.7009304
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.7044887, upper bound: 1.6946513
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.6976511, upper bound: 1.7014853
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.7039318, upper bound: 1.6952050
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.6907058, upper bound: 1.7084310
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.6969865, upper bound: 1.7021511
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.6901489, upper bound: 1.7089883
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.6964296, upper bound: 1.7027078
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.7027080, upper bound: 1.6964297
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 26.60
Output dim: 8, lower bound: -1.7089882, upper bound: 1.6901489
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.60
Output dim: 8, lower bound: -1.7084412, upper bound: 1.6969966
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.60
Output dim: 8, lower bound: -1.7014953, upper bound: 1.7039419
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.60
Output dim: 8, lower bound: -1.7009408, upper bound: 1.7044989
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=2.5866003036499023
rel_dist={8: [-1.7090580688748975, 1.709057719074428]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966896, upper bound: 1.4969374
time: 10.01 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4969355, upper bound: 1.4966915
time: 5.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.49
Output dim: 8, lower bound: -1.4966896, upper bound: 1.4969374
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.49
Output dim: 8, lower bound: -1.4969355, upper bound: 1.4966915

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5843282, 2.5810165
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7332797, 2.7312436
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.4074249, 2.4061394
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7498775, 2.7541595
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5641122, 1.5672255
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3543200, 2.3601859
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4352093, 2.4352269
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9730077, 2.9699860

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4556

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4931505, upper bound: 1.4969212
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966734, upper bound: 1.4933961
time: 7.44 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5810170, 2.5827990
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7312427, 2.7323341
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.4061394, 2.4068317
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7521796, 2.7498775
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5657897, 1.5641122
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3574739, 2.3543198
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4352207, 2.4352088
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9699855, 2.9716167

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4556

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4933965, upper bound: 1.4966734
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4969193, upper bound: 1.4931500
time: 6.01 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 24.80 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.80
Output dim: 8, lower bound: -1.4931505, upper bound: 1.4969212
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.80
Output dim: 8, lower bound: -1.4966734, upper bound: 1.4933961
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 24.80
Output dim: 8, lower bound: -1.4933965, upper bound: 1.4966734
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 24.80
Output dim: 8, lower bound: -1.4969193, upper bound: 1.4931500

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5810127, 2.5814738
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7380099, 2.7308722
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3847446, 2.3943338
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7377229, 2.7269444
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5451143, 1.5396709
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3408022, 2.3386643
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4277229, 2.4299269
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9595466, 2.9604468

Time for backsubstitution: 12.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4931408, upper bound: 1.4911247
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4873559, upper bound: 1.4969087
time: 5.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5847864, 2.5777011
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7329087, 2.7359734
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3956184, 2.3834596
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7226624, 2.7420051
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5365579, 1.5482273
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3327980, 2.3466685
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4299078, 2.4277415
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9634681, 2.9565239

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966629, upper bound: 1.4876015
time: 7.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4908787, upper bound: 1.4933863
time: 6.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5777006, 2.5832562
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7359738, 2.7319622
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3834591, 2.3950262
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7400260, 2.7226624
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5467904, 1.5365577
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3439565, 2.3327982
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4277334, 2.4299083
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9565234, 2.9620776

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4933867, upper bound: 1.4908786
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4876018, upper bound: 1.4966627
time: 7.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5814734, 2.5794835
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7308717, 2.7370634
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3943338, 2.3841519
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7249656, 2.7377234
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5382340, 1.5451140
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3359523, 2.3408024
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4299192, 2.4277229
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9604468, 2.9581547

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4969088, upper bound: 1.4873556
time: 6.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4911247, upper bound: 1.4931408
time: 5.58 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 25.00 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.00
Output dim: 8, lower bound: -1.4931408, upper bound: 1.4911247
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.00
Output dim: 8, lower bound: -1.4873559, upper bound: 1.4969087
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.00
Output dim: 8, lower bound: -1.4966629, upper bound: 1.4876015
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.00
Output dim: 8, lower bound: -1.4908787, upper bound: 1.4933863
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.00
Output dim: 8, lower bound: -1.4933867, upper bound: 1.4908786
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.00
Output dim: 8, lower bound: -1.4876018, upper bound: 1.4966627
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 25.00
Output dim: 8, lower bound: -1.4969088, upper bound: 1.4873556
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 25.00
Output dim: 8, lower bound: -1.4911247, upper bound: 1.4931408

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5805192, 2.5797825
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7371335, 2.7306089
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3836088, 2.3904362
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7187591, 2.7214255
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5419660, 1.5287333
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3400745, 2.3361597
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4248204, 2.4198627
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9460521, 2.9565639

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4931228, upper bound: 1.4905380
time: 6.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4925543, upper bound: 1.4911085
time: 12.44 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5793214, 2.5809798
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7377477, 2.7299948
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3808470, 2.3931980
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7322078, 2.7079802
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5341763, 1.5365245
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3382978, 2.3379364
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4176593, 2.4270248
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9556632, 2.9469547

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4873379, upper bound: 1.4963224
time: 7.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4867694, upper bound: 1.4968912
time: 5.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5842919, 2.5760095
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7320313, 2.7357116
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3944836, 2.3795624
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2425356, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7036977, 2.7364893
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5334115, 1.5372896
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3320704, 2.3441639
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4270062, 2.4176774
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9499755, 2.9526420

Time for backsubstitution: 12.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966450, upper bound: 1.4870152
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4960765, upper bound: 1.4875837
time: 6.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5830941, 2.5772076
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7326455, 2.7350969
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3917217, 2.3823237
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7171435, 2.7230408
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5256200, 1.5450792
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3302937, 2.3459404
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4198442, 2.4248390
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9595866, 2.9430313

Time for backsubstitution: 12.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4908608, upper bound: 1.4927998
time: 7.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4902923, upper bound: 1.4933687
time: 5.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5772080, 2.5815639
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7350965, 2.7316995
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3823242, 2.3911281
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7210612, 2.7171438
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5436420, 1.5256200
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3432279, 2.3302934
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4248309, 2.4198442
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9430308, 2.9581947

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4933687, upper bound: 1.4902920
time: 7.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4928002, upper bound: 1.4908608
time: 6.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5760093, 2.5827613
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7357116, 2.7310853
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3795624, 2.3938899
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2425356
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7345099, 2.7036982
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5358524, 1.5334113
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3414512, 2.3320701
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4176688, 2.4270062
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9526420, 2.9485855

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4875838, upper bound: 1.4960764
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4870153, upper bound: 1.4966447
time: 6.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5809798, 2.5777910
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7299943, 2.7368021
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3931980, 2.3802538
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2432137, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7060008, 2.7322075
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5350876, 1.5341763
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3352237, 2.3382976
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4270167, 2.4176588
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9469543, 2.9542727

Time for backsubstitution: 14.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4968909, upper bound: 1.4867696
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4963224, upper bound: 1.4873377
time: 6.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5797830, 2.5789890
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7306085, 2.7361870
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3904362, 2.3830156
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7194467, 2.7187591
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5272961, 1.5419660
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3334470, 2.3400743
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4198546, 2.4248204
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9565635, 2.9446621

Time for backsubstitution: 14.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4911067, upper bound: 1.4925541
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4905382, upper bound: 1.4931227
time: 6.69 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 27.34 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.34
Output dim: 8, lower bound: -1.4931228, upper bound: 1.4905380
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.34
Output dim: 8, lower bound: -1.4925543, upper bound: 1.4911085
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.34
Output dim: 8, lower bound: -1.4873379, upper bound: 1.4963224
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.34
Output dim: 8, lower bound: -1.4867694, upper bound: 1.4968912
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.34
Output dim: 8, lower bound: -1.4966450, upper bound: 1.4870152
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.34
Output dim: 8, lower bound: -1.4960765, upper bound: 1.4875837
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.34
Output dim: 8, lower bound: -1.4908608, upper bound: 1.4927998
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.34
Output dim: 8, lower bound: -1.4902923, upper bound: 1.4933687
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.34
Output dim: 8, lower bound: -1.4933687, upper bound: 1.4902920
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.34
Output dim: 8, lower bound: -1.4928002, upper bound: 1.4908608
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.34
Output dim: 8, lower bound: -1.4875838, upper bound: 1.4960764
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.34
Output dim: 8, lower bound: -1.4870153, upper bound: 1.4966447
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.34
Output dim: 8, lower bound: -1.4968909, upper bound: 1.4867696
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.34
Output dim: 8, lower bound: -1.4963224, upper bound: 1.4873377
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.34
Output dim: 8, lower bound: -1.4911067, upper bound: 1.4925541
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.34
Output dim: 8, lower bound: -1.4905382, upper bound: 1.4931227

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5862231, 2.5762224
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7388067, 2.7295637
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3805752, 2.3952909
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7260842, 2.7168298
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5467715, 1.5257342
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3397961, 2.3366010
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4236732, 2.4217243
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9531264, 2.9521441

Time for backsubstitution: 12.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4882329, upper bound: 1.4905303
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4931145, upper bound: 1.4856509
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5769601, 2.5797825
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7360868, 2.7306089
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3836088, 2.3874035
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7141633, 2.7214255
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5389671, 1.5287333
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3400745, 2.3358817
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4248204, 2.4187155
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9416337, 2.9565639

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4876644, upper bound: 1.4910989
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4925460, upper bound: 1.4862194
time: 5.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5850253, 2.5774200
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7394209, 2.7289495
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3778143, 2.3980522
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2433224
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7395329, 2.7033844
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5389814, 1.5335252
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3380194, 2.3383777
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4165120, 2.4288864
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9627366, 2.9425344

Time for backsubstitution: 12.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4824480, upper bound: 1.4963162
time: 7.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4873296, upper bound: 1.4914350
time: 5.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5757613, 2.5809798
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7367029, 2.7299948
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3808470, 2.3901649
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7276120, 2.7079802
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5311770, 1.5365245
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3382978, 2.3376584
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4176593, 2.4258776
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9512439, 2.9469547

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4818795, upper bound: 1.4968827
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4867611, upper bound: 1.4920035
time: 5.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5899959, 2.5724497
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7337046, 2.7346659
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3914499, 2.3844166
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2432985, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7110238, 2.7318935
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5382166, 1.5342903
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3317919, 2.3446052
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4258590, 2.4195395
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9570498, 2.9482222

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4917557, upper bound: 1.4870089
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966369, upper bound: 1.4821273
time: 5.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5807319, 2.5760095
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7309866, 2.7357116
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3944836, 2.3765292
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2420645, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6991019, 2.7364893
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5304117, 1.5372896
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3320704, 2.3438859
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4270062, 2.4165306
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9455562, 2.9526420

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4911872, upper bound: 1.4875752
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4960684, upper bound: 1.4826936
time: 7.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5887980, 2.5736475
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7343187, 2.7340508
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3886890, 2.3871779
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7244687, 2.7184451
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5304255, 1.5420802
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3300152, 2.3463819
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4186969, 2.4267006
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9666600, 2.9386115

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4859716, upper bound: 1.4927938
time: 10.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4908527, upper bound: 1.4879103
time: 8.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5795341, 2.5772076
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7316008, 2.7350969
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3917217, 2.3792911
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7125478, 2.7230408
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5226212, 1.5450792
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3302937, 2.3456624
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4198442, 2.4236917
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9551663, 2.9430313

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4854031, upper bound: 1.4933623
time: 5.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4902842, upper bound: 1.4884787
time: 6.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5829110, 2.5780044
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7367697, 2.7306547
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3792915, 2.3959823
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7283874, 2.7125480
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5484481, 1.5226209
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3429503, 2.3307350
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4236846, 2.4217062
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9501052, 2.9537749

Time for backsubstitution: 12.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4884788, upper bound: 1.4902840
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4933604, upper bound: 1.4854035
time: 6.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5736470, 2.5815639
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7340508, 2.7316995
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3823242, 2.3880954
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7164664, 2.7171438
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5406432, 1.5256200
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3432279, 2.3300154
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4248309, 2.4186974
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9386115, 2.9581947

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4879103, upper bound: 1.4908547
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4927920, upper bound: 1.4859735
time: 4.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5817132, 2.5792022
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7373838, 2.7300406
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3765297, 2.3987441
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2420645
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7418351, 2.6991024
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5406585, 1.5304120
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3411736, 2.3325114
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4165235, 2.4288683
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9597154, 2.9441652

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4826939, upper bound: 1.4960703
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4875755, upper bound: 1.4911876
time: 5.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5724492, 2.5827613
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7346659, 2.7310853
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3795624, 2.3908567
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2425356
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7299142, 2.7036982
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5328536, 1.5334113
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3414512, 2.3317921
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4176688, 2.4258595
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9482217, 2.9485855

Time for backsubstitution: 12.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4821254, upper bound: 1.4966389
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4870070, upper bound: 1.4917553
time: 7.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5866838, 2.5742316
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7316675, 2.7357569
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3901653, 2.3851085
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7133260, 2.7276118
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5398927, 1.5311770
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3349462, 2.3387392
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4258704, 2.4195204
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9540277, 2.9498529

Time for backsubstitution: 12.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4920016, upper bound: 1.4867631
time: 5.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4968828, upper bound: 1.4818815
time: 7.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5774198, 2.5777910
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7289495, 2.7368021
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3931980, 2.3772211
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2427425, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7014050, 2.7322075
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5320878, 1.5341763
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3352237, 2.3380196
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4270167, 2.4165120
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9425340, 2.9542727

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4914331, upper bound: 1.4873316
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4963143, upper bound: 1.4824500
time: 4.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5854859, 2.5754297
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7322817, 2.7351422
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3874035, 2.3878703
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7267718, 2.7141633
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5321021, 1.5389669
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3331695, 2.3405156
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4187083, 2.4266825
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9636378, 2.9402423

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4862175, upper bound: 1.4925480
time: 5.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4910986, upper bound: 1.4876664
time: 5.16 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 23.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4882329, upper bound: 1.4905303
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4931145, upper bound: 1.4856509
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4876644, upper bound: 1.4910989
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4925460, upper bound: 1.4862194
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4824480, upper bound: 1.4963162
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4873296, upper bound: 1.4914350
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4818795, upper bound: 1.4968827
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4867611, upper bound: 1.4920035
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4917557, upper bound: 1.4870089
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4966369, upper bound: 1.4821273
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4911872, upper bound: 1.4875752
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4960684, upper bound: 1.4826936
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4859716, upper bound: 1.4927938
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4908527, upper bound: 1.4879103
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4854031, upper bound: 1.4933623
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4902842, upper bound: 1.4884787
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4884788, upper bound: 1.4902840
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4933604, upper bound: 1.4854035
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4879103, upper bound: 1.4908547
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4927920, upper bound: 1.4859735
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4826939, upper bound: 1.4960703
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4875755, upper bound: 1.4911876
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4821254, upper bound: 1.4966389
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4870070, upper bound: 1.4917553
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4920016, upper bound: 1.4867631
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4968828, upper bound: 1.4818815
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4914331, upper bound: 1.4873316
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4963143, upper bound: 1.4824500
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4862175, upper bound: 1.4925480
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 23.46
Output dim: 8, lower bound: -1.4910986, upper bound: 1.4876664
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.46
Output dim: 8, lower bound: -1.4905382, upper bound: 1.4931227
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=2.435220718383789
rel_dist={8: [-1.4969366874007388, 1.496938647628781]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3844229, upper bound: 1.3847677
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847674, upper bound: 1.3844231
time: 4.39 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.77 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.77
Output dim: 8, lower bound: -1.3844229, upper bound: 1.3847677
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.77
Output dim: 8, lower bound: -1.3847674, upper bound: 1.3844231

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4982009, 2.4953618
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6564956, 2.6547503
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3245182, 2.3234167
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2016230, 3.2027006
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6281104, 2.6317801
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5009766, 1.5036452
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2614970, 2.2665250
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3595181, 2.3595343
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8934798, 2.8908901

Time for backsubstitution: 12.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4556

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3813645, upper bound: 1.3847557
time: 7.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3844092, upper bound: 1.3817093
time: 4.12 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4953618, 2.4971445
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6547503, 2.6558409
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3234167, 2.3241086
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2023001, 3.2016220
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6304116, 2.6281102
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5026536, 1.5009766
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2646513, 2.2614970
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3595295, 2.3595185
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8908896, 2.8925209

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4556

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3817091, upper bound: 1.3844092
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847536, upper bound: 1.3813648
time: 4.23 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 22.47 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.47
Output dim: 8, lower bound: -1.3813645, upper bound: 1.3847557
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.47
Output dim: 8, lower bound: -1.3844092, upper bound: 1.3817093
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 22.47
Output dim: 8, lower bound: -1.3817091, upper bound: 1.3844092
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 22.47
Output dim: 8, lower bound: -1.3847536, upper bound: 1.3813648

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4948854, 2.4952803
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6604972, 2.6543789
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3018384, 2.3100572
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1878920, 3.1843896
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6138043, 2.6045651
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4807560, 1.4760907
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2468367, 2.2450035
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3520327, 2.3539219
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8800178, 2.8807907

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3813558, upper bound: 1.3798089
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3764194, upper bound: 1.3847441
time: 9.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4981194, 2.4920464
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6561246, 2.6587515
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3111587, 2.3007369
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1833105, 3.1889710
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6008954, 2.6174746
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4734223, 1.4834247
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2399759, 2.2518642
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3539057, 2.3520484
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8833804, 2.8774281

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3843998, upper bound: 1.3767645
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3794642, upper bound: 1.3817006
time: 4.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4920464, 2.4970627
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6587510, 2.6554689
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3007369, 2.3107495
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1885710, 3.1833115
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6161075, 2.6008952
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4824326, 1.4734221
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2499900, 2.2399755
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3520432, 2.3539062
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8774276, 2.8824215

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3817003, upper bound: 1.3794640
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3767642, upper bound: 1.3844001
time: 5.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4952803, 2.4938288
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6543784, 2.6598415
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3100572, 2.3014288
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1839895, 3.1878924
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6031976, 2.6138043
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4750988, 1.4807563
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2431293, 2.2468362
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3539162, 2.3520327
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8807902, 2.8790588

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847443, upper bound: 1.3764197
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3798089, upper bound: 1.3813561
time: 4.30 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 21.16
Output dim: 8, lower bound: -1.3813558, upper bound: 1.3798089
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 8, lower bound: -1.3764194, upper bound: 1.3847441
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 8, lower bound: -1.3843998, upper bound: 1.3767645
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 21.16
Output dim: 8, lower bound: -1.3794642, upper bound: 1.3817006
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 21.16
Output dim: 8, lower bound: -1.3817003, upper bound: 1.3794640
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 8, lower bound: -1.3767642, upper bound: 1.3844001
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.16
Output dim: 8, lower bound: -1.3847443, upper bound: 1.3764197
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 21.16
Output dim: 8, lower bound: -1.3798089, upper bound: 1.3813561

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4931931, 2.4946153
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6601472, 2.6535020
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.2979407, 2.3085270
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1842289, 3.1750178
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6063676, 2.5856009
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4698186, 1.4718311
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2443314, 2.2440217
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3419681, 2.3499966
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8747630, 2.8672981

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3764040, upper bound: 1.3841560
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3758292, upper bound: 1.3847285
time: 6.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4974542, 2.4903550
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6552472, 2.6584020
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3096285, 2.2968392
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1739397, 3.1853070
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.5819306, 2.6100373
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4691625, 1.4724870
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2389936, 2.2493596
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3499808, 2.3419843
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8698878, 2.8721728

Time for backsubstitution: 12.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3843845, upper bound: 1.3761739
time: 8.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3838095, upper bound: 1.3767486
time: 11.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4903550, 2.4963968
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6584020, 2.6545920
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.2968392, 2.3092189
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1849070, 3.1739392
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6086698, 2.5819309
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4714947, 1.4691625
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2474852, 2.2389936
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3419795, 2.3499808
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8721728, 2.8689289

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3767487, upper bound: 1.3838095
time: 12.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3761739, upper bound: 1.3843864
time: 5.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4946151, 2.4921365
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6535020, 2.6594920
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3085270, 2.2975311
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1746178, 3.1842289
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.5842338, 2.6063673
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4708385, 1.4698186
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2421470, 2.2443316
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3499913, 2.3419685
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8672976, 2.8738036

Time for backsubstitution: 12.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847289, upper bound: 1.3758288
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3841539, upper bound: 1.3764043
time: 4.47 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 24.35 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.35
Output dim: 8, lower bound: -1.3764040, upper bound: 1.3841560
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.35
Output dim: 8, lower bound: -1.3758292, upper bound: 1.3847285
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.35
Output dim: 8, lower bound: -1.3843845, upper bound: 1.3761739
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.35
Output dim: 8, lower bound: -1.3838095, upper bound: 1.3767486
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 24.35
Output dim: 8, lower bound: -1.3767487, upper bound: 1.3838095
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 24.35
Output dim: 8, lower bound: -1.3761739, upper bound: 1.3843864
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 24.35
Output dim: 8, lower bound: -1.3847289, upper bound: 1.3758288
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 24.35
Output dim: 8, lower bound: -1.3841539, upper bound: 1.3764043

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4896340, 2.4946153
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6591024, 2.6535020
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.2979407, 2.3054938
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1837568, 3.1750178
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6017718, 2.5856009
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4668193, 1.4718311
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2443314, 2.2437437
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3419681, 2.3488498
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8703427, 2.8672981

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3716340, upper bound: 1.3847223
time: 6.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3758222, upper bound: 1.3805451
time: 6.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5018344, 2.4867949
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6565313, 2.6573563
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3065953, 2.3005672
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1745262, 3.1848359
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.5875525, 2.6054416
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4728532, 1.4694877
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2387161, 2.2496982
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3488345, 2.3434167
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8753200, 2.8677530

Time for backsubstitution: 12.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3802008, upper bound: 1.3761669
time: 8.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3843776, upper bound: 1.3719783
time: 13.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4867949, 2.4963968
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6573553, 2.6545920
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.2968392, 2.3061857
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1844358, 3.1739392
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6040750, 2.5819309
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4684958, 1.4691625
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2474852, 2.2387156
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3419795, 2.3488340
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8677526, 2.8689289

Time for backsubstitution: 12.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3719787, upper bound: 1.3843773
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3761668, upper bound: 1.3802010
time: 6.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4989953, 2.4885771
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6547861, 2.6584473
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3054938, 2.3012586
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1752052, 3.1837573
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.5898557, 2.6017716
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4745288, 1.4668193
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2418694, 2.2446702
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3488450, 2.3434010
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8727298, 2.8693838

Time for backsubstitution: 12.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3805454, upper bound: 1.3758219
time: 7.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847220, upper bound: 1.3716340
time: 6.81 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 27.35 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.35
Output dim: 8, lower bound: -1.3716340, upper bound: 1.3847223
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.35
Output dim: 8, lower bound: -1.3758222, upper bound: 1.3805451
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 27.35
Output dim: 8, lower bound: -1.3802008, upper bound: 1.3761669
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.35
Output dim: 8, lower bound: -1.3843776, upper bound: 1.3719783
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 27.35
Output dim: 8, lower bound: -1.3719787, upper bound: 1.3843773
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.35
Output dim: 8, lower bound: -1.3761668, upper bound: 1.3802010
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 27.35
Output dim: 8, lower bound: -1.3805454, upper bound: 1.3758219
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.35
Output dim: 8, lower bound: -1.3847220, upper bound: 1.3716340

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5229979, 2.5202565
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6315107, 2.6110840
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.2603025, 2.2772551
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1594534, 3.1567793
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.5554290, 2.5238323
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4576726, 1.4596484
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2298455, 2.2244325
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3276353, 2.3407297
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8555083, 2.8475270

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5832

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3678525, upper bound: 1.3847184
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3716303, upper bound: 1.3809269
time: 9.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5246363, 2.5219407
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6123686, 2.6308551
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.2772551, 2.2636223
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1569662, 3.1594539
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.5280881, 2.5554290
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4623466, 1.4576726
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2225585, 2.2301838
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3407264, 2.3290677
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8529592, 2.8545489

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 5832

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3809273, upper bound: 1.3716302
time: 5.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847183, upper bound: 1.3678519
time: 14.39 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 33.01 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 33.01
Output dim: 8, lower bound: -1.3678525, upper bound: 1.3847184
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 33.01
Output dim: 8, lower bound: -1.3716303, upper bound: 1.3809269
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 33.01
Output dim: 8, lower bound: -1.3809273, upper bound: 1.3716302
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 33.01
Output dim: 8, lower bound: -1.3847183, upper bound: 1.3678519

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5220432, 2.5198345
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6278558, 2.6026993
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.2571945, 2.2758961
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1587429, 3.1564684
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.5472875, 2.5202746
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4561424, 1.4589787
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2273140, 2.2186346
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3233867, 2.3388748
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8523331, 2.8402491

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 918

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3676045, upper bound: 1.3845243
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 4671

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3646265, upper bound: 1.3847134
time: 5.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3678477, upper bound: 1.3814647
time: 12.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5242138, 2.5209870
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6039844, 2.6272006
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.2758961, 2.2605128
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1566572, 3.1587429
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.5245309, 2.5472870
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4616771, 1.4561424
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2167602, 2.2276528
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3388705, 2.3248196
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8456802, 2.8513751

Time for backsubstitution: 13.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 918

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3806972, upper bound: 1.3675936
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3845248, upper bound: 1.3676066
time: 5.93 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 25.32 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 25.32
Output dim: 8, lower bound: -1.3646265, upper bound: 1.3847134
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 25.32
Output dim: 8, lower bound: -1.3678477, upper bound: 1.3814647
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 25.32
Output dim: 8, lower bound: -1.3806972, upper bound: 1.3675936
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 25.32
Output dim: 8, lower bound: -1.3845248, upper bound: 1.3676066

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5158405, 2.5052552
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6175995, 2.5785384
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.2458429, 2.2492285
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1456461, 3.1509080
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.5442405, 2.5131302
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4464860, 1.4548807
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2221684, 2.2065325
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3159795, 2.3357263
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8520737, 2.8401327

Time for backsubstitution: 12.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 918

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3643759, upper bound: 1.3845219
time: 5.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3643668, upper bound: 1.3806923
time: 6.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5242138, 2.5208230
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6025801, 2.6272006
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.2758961, 2.2601376
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1566257, 3.1587429
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.5245309, 2.5460856
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4616771, 1.4557581
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2158456, 2.2276528
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3388705, 2.3239603
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8447905, 2.8513751

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 4671

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3812704, upper bound: 1.3675996
time: 6.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3845199, upper bound: 1.3643759
time: 5.41 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 24.48 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 24.48
Output dim: 8, lower bound: -1.3643759, upper bound: 1.3845219
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 24.48
Output dim: 8, lower bound: -1.3643668, upper bound: 1.3806923
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 24.48
Output dim: 8, lower bound: -1.3812704, upper bound: 1.3675996
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 24.48
Output dim: 8, lower bound: -1.3845199, upper bound: 1.3643759

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5156765, 2.5052810
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6176815, 2.5771341
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.2454677, 2.2492509
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1456604, 3.1508780
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.5430388, 2.5132036
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4461017, 1.4549038
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2222252, 2.2056179
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3151212, 2.3357763
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8521271, 2.8392420

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 822

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3504683, upper bound: 1.3840316
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3638852, upper bound: 1.3706115
time: 6.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5096359, 2.5146208
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.5784187, 2.6169438
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.2492285, 2.2487860
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1510658, 3.1456456
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.5173860, 2.5430386
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4575791, 1.4461014
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2037435, 2.2225070
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3357215, 2.3165531
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8446732, 2.8511152

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 822

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3706115, upper bound: 1.3638845
time: 6.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3840296, upper bound: 1.3504683
time: 6.39 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 25.73 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 25.73
Output dim: 8, lower bound: -1.3504683, upper bound: 1.3840316
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 25.73
Output dim: 8, lower bound: -1.3638852, upper bound: 1.3706115
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 25.73
Output dim: 8, lower bound: -1.3706115, upper bound: 1.3638845
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 25.73
Output dim: 8, lower bound: -1.3840296, upper bound: 1.3504683
Binary search (step 2): status=Status.VERIFIED, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=2.359529972076416
rel_dist={8: [-1.3847687820365167, 1.3847690605870273]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 2275.16 seconds
