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
execution time: IAR + LP analysis = 13.07 + 57.87 = 70.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -1.9845428, upper bound: 1.9845425


# Binary Search by BASE starts (time budget: 3529.06 seconds, max iter: 100)

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
Binary search time: 194.09 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Relational Split (RS_random_Z) starts
Time budget: 3334.97 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6124

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7057681, upper bound: 1.7090563
time: 5.79 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7090564, upper bound: 1.7057678
time: 6.93 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.73
Output dim: 8, lower bound: -1.7057681, upper bound: 1.7090563
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.73
Output dim: 8, lower bound: -1.7090564, upper bound: 1.7057678

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8841085, 2.8778968
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5613184, 2.5704982
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9945803, 2.9887323
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6907921, 1.6842334
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5421495, 2.5371244
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5836105, 2.5861101
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1296735, 3.1289959

Time for backsubstitution: 12.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5788

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7057550, upper bound: 1.7015405
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6982555, upper bound: 1.7090435
time: 5.13 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8778963, 2.8841090
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5704985, 2.5613189
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9887323, 2.9945807
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6842332, 1.6907923
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5371246, 2.5421495
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5861101, 2.5836110
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1289954, 3.1296730

Time for backsubstitution: 12.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7027658, upper bound: 1.7057580
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7090464, upper bound: 1.6994783
time: 5.76 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.65 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.65
Output dim: 8, lower bound: -1.7057550, upper bound: 1.7015405
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.65
Output dim: 8, lower bound: -1.6982555, upper bound: 1.7090435
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.65
Output dim: 8, lower bound: -1.7027658, upper bound: 1.7057580
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.65
Output dim: 8, lower bound: -1.7090464, upper bound: 1.6994783

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8832312, 2.8778095
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5609741, 2.5666025
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9756212, 2.9870596
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6898708, 1.6732953
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5419292, 2.5346196
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5827551, 2.5760469
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1161828, 3.1278615

Time for backsubstitution: 13.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4556

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 822

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6834154, upper bound: 1.7012220
time: 6.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7054362, upper bound: 1.6791998
time: 6.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8840189, 2.8770199
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5574226, 2.5701532
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9929066, 2.9697728
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6798544, 1.6833124
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5396447, 2.5369039
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5735474, 2.5852537
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1285386, 3.1155052

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 822

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6759156, upper bound: 1.7087249
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6979369, upper bound: 1.6867032
time: 5.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8577170, 2.8416924
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5328569, 2.5377822
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9501114, 2.9328136
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6766160, 1.6786149
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5250492, 2.5228367
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5717802, 2.5786009
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1166248, 3.1098986

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5832

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6975473, upper bound: 1.7057521
time: 5.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7027602, upper bound: 1.7005106
time: 8.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8354802, 2.8639302
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5469618, 2.5236781
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9269648, 2.9559596
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6720564, 1.6831743
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5178113, 2.5300746
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5811005, 2.5692811
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1092205, 3.1173024

Time for backsubstitution: 12.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 4671

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4556

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7045246, upper bound: 1.6994587
time: 7.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7090252, upper bound: 1.6949467
time: 7.06 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.02 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 8, lower bound: -1.6834154, upper bound: 1.7012220
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 8, lower bound: -1.7054362, upper bound: 1.6791998
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 8, lower bound: -1.6759156, upper bound: 1.7087249
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 8, lower bound: -1.6979369, upper bound: 1.6867032
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 8, lower bound: -1.6975473, upper bound: 1.7057521
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 8, lower bound: -1.7027602, upper bound: 1.7005106
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 8, lower bound: -1.7045246, upper bound: 1.6994587
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 8, lower bound: -1.7090252, upper bound: 1.6949467

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8675632, 2.8527250
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5653286, 2.5694075
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9760261, 2.9891603
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6887779, 1.6715522
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5308237, 2.5168345
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5752664, 2.5713682
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1162052, 3.1277075

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6763649, upper bound: 1.7012081
time: 6.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6833991, upper bound: 1.6941754
time: 7.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8581467, 2.8621407
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5637789, 2.5709579
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9777217, 2.9874640
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6881285, 1.6722022
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5241442, 2.5235140
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5780768, 2.5685582
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1160278, 3.1278844

Time for backsubstitution: 12.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 4556

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6795084, upper bound: 1.6754170
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7054153, upper bound: 1.6752916
time: 6.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8683510, 2.8519354
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5617790, 2.5729582
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9933114, 2.9718733
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6787605, 1.6815693
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5285392, 2.5191188
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5660586, 2.5805755
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1285629, 3.1153507

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 4671

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6696248, upper bound: 1.7087143
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6759054, upper bound: 1.7024343
time: 5.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8589344, 2.8613515
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5602283, 2.5745087
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9950070, 2.9701772
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6781111, 1.6822193
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5218596, 2.5257983
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5688691, 2.5777650
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1283855, 3.1155281

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6916462, upper bound: 1.6866928
time: 8.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6979268, upper bound: 1.6804126
time: 5.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8564267, 2.8333087
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5297489, 2.5372984
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9419689, 2.9315495
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6750858, 1.6783751
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5241513, 2.5170388
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5675306, 2.5779419
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1155024, 3.1026196

Time for backsubstitution: 12.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4556

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6930252, upper bound: 1.7057345
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6975266, upper bound: 1.7012243
time: 6.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8493333, 2.8404031
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5323744, 2.5346732
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9488468, 2.9246716
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6763761, 1.6770847
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5192513, 2.5219390
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5711212, 2.5743513
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1093454, 3.1087756

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 822

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6804201, upper bound: 1.7001930
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7024416, upper bound: 1.6781750
time: 6.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8416662, 2.8635573
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5242844, 2.5149822
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9191031, 2.9287343
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6554980, 1.6556153
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5065737, 2.5085456
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5736094, 2.5646000
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.0957632, 3.1088910

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 5832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4671

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7001494, upper bound: 1.6994518
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7045179, upper bound: 1.6950221
time: 5.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8351068, 2.8701134
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5382376, 2.5010011
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.8997397, 2.9480624
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6444969, 1.6665938
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.4962826, 2.5188224
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5764160, 2.5617905
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1008072, 3.1038446

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 5788

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6837455, upper bound: 1.6949277
time: 6.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7090055, upper bound: 1.6696691
time: 6.92 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.50
Output dim: 8, lower bound: -1.6763649, upper bound: 1.7012081
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.50
Output dim: 8, lower bound: -1.6833991, upper bound: 1.6941754
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.50
Output dim: 8, lower bound: -1.6795084, upper bound: 1.6754170
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.50
Output dim: 8, lower bound: -1.7054153, upper bound: 1.6752916
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.50
Output dim: 8, lower bound: -1.6696248, upper bound: 1.7087143
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.50
Output dim: 8, lower bound: -1.6759054, upper bound: 1.7024343
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.50
Output dim: 8, lower bound: -1.6916462, upper bound: 1.6866928
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.50
Output dim: 8, lower bound: -1.6979268, upper bound: 1.6804126
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.50
Output dim: 8, lower bound: -1.6930252, upper bound: 1.7057345
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.50
Output dim: 8, lower bound: -1.6975266, upper bound: 1.7012243
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.50
Output dim: 8, lower bound: -1.6804201, upper bound: 1.7001930
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.50
Output dim: 8, lower bound: -1.7024416, upper bound: 1.6781750
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.50
Output dim: 8, lower bound: -1.7001494, upper bound: 1.6994518
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.50
Output dim: 8, lower bound: -1.7045179, upper bound: 1.6950221
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.50
Output dim: 8, lower bound: -1.6837455, upper bound: 1.6949277
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.50
Output dim: 8, lower bound: -1.7090055, upper bound: 1.6696691

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8650579, 2.8366990
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5412807, 2.5656762
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9734516, 2.9728143
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6836352, 1.6382827
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5236621, 2.4704156
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5673890, 2.5701365
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1069641, 3.1262770

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6763418, upper bound: 1.7006298
time: 7.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6757793, upper bound: 1.7011851
time: 9.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8515368, 2.8502159
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5615921, 2.5453587
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9596796, 2.9865880
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6555085, 1.6664077
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.4844050, 2.5096729
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5740323, 2.5634909
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1147747, 3.1184664

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5788

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5832

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6781556, upper bound: 1.6941702
time: 7.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6833936, upper bound: 1.6889610
time: 5.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8061752, 2.7769141
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5650387, 2.5718513
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9926882, 3.0085664
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6639662, 1.6335316
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.4815416, 2.4478602
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5185452, 2.5313444
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1264954, 3.1434555

Time for backsubstitution: 12.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 5788

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6732178, upper bound: 1.6754069
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6794984, upper bound: 1.6691258
time: 5.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7729206, 2.8101649
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5646725, 2.5722175
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9988241, 3.0024309
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6494579, 1.6480430
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.4484901, 2.4809031
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5408659, 2.5090272
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1319532, 3.1383514

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 4671

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6991250, upper bound: 1.6752817
time: 6.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7054053, upper bound: 1.6690025
time: 6.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8481722, 2.8095198
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5241375, 2.5494213
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9546893, 2.9101055
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6711431, 1.6693919
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5164638, 2.4998055
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5517287, 2.5755649
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1161942, 3.0955796

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4671

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6652195, upper bound: 1.7087098
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6696182, upper bound: 1.7043489
time: 5.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8259354, 2.8317561
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5382414, 2.5353174
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9315436, 2.9332516
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6665835, 1.6739514
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5092258, 2.5070431
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5610490, 2.5662451
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1087899, 3.1029835

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 4556

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 918

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6716119, upper bound: 1.7008275
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6742989, upper bound: 1.6981406
time: 6.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8387566, 2.8189354
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5225868, 2.5509715
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9563849, 2.9084094
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6704926, 1.6700418
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5097842, 2.5064850
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5545392, 2.5727549
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1160169, 3.0957565

Time for backsubstitution: 12.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6657179, upper bound: 1.6829103
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6916254, upper bound: 1.6827847
time: 8.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8165188, 2.8411722
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5366917, 2.5368679
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9332392, 2.9315555
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6659331, 1.6746013
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5025463, 2.5137227
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5638585, 2.5634351
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1086144, 3.1031609

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4671

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6935150, upper bound: 1.6804060
time: 7.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6979202, upper bound: 1.6760531
time: 6.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8626175, 2.8329353
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5070715, 2.5285957
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9340978, 2.9043238
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6585221, 1.6508164
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5129132, 2.4955099
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5600414, 2.5732617
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1020441, 3.0942063

Time for backsubstitution: 12.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 4671

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6930130, upper bound: 1.6982198
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6855106, upper bound: 1.7057214
time: 5.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8560543, 2.8394890
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5210314, 2.5146215
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9147439, 2.9236615
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6475267, 1.6618003
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5026221, 2.5057871
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5628481, 2.5704517
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1070910, 3.0891619

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6975033, upper bound: 1.7006347
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6969458, upper bound: 1.7012010
time: 8.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8336639, 2.8153176
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5367308, 2.5374794
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9492519, 2.9267726
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6752832, 1.6753418
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5081453, 2.5041535
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5636330, 2.5696726
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1093721, 3.1086245

Time for backsubstitution: 12.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 4556

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 918

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6750381, upper bound: 1.6999947
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6788131, upper bound: 1.7000164
time: 6.54 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8242483, 2.8247337
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5351801, 2.5390296
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9509485, 2.9250765
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6746333, 1.6759918
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5014658, 2.5108330
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5664425, 2.5668631
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1091948, 3.1088018

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7024184, upper bound: 1.6775875
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7018615, upper bound: 1.6781518
time: 7.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8383603, 2.8393955
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5205908, 2.4883144
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9181058, 2.9215889
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6458421, 1.6542968
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.5049028, 2.4964426
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5662022, 2.5635796
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.0955038, 3.1088476

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 5832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7001469, upper bound: 1.6994495
time: 5.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7001469, upper bound: 1.6994495
time: 5.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.8175044, 2.8602505
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.4976168, 2.5112879
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9119585, 2.9277360
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6541800, 1.6459589
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.4944706, 2.5068743
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5725889, 2.5571928
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.0957193, 3.1086321

Time for backsubstitution: 15.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5830

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7044946, upper bound: 1.6944346
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.7039377, upper bound: 1.6949996
time: 5.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7797813, 2.7815323
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5394955, 2.5018935
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9120424, 2.9665005
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6203370, 1.6279244
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.4415965, 2.4310846
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5168819, 2.5245728
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1115928, 3.1200876

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 918

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6794519, upper bound: 1.6933115
time: 8.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6821390, upper bound: 1.6906034
time: 6.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6798801, 2.6798801
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7465258, 2.8147869
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.5391293, 2.5022595
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.9181774, 2.9603651
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.6058278, 1.6424334
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.4085445, 2.4641361
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.5391989, 2.5022559
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -3.1170497, 3.1146302

Time for backsubstitution: 12.38 seconds
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
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5830

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966896, upper bound: 1.4969374
time: 10.04 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4969355, upper bound: 1.4966915
time: 5.33 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.39 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.39
Output dim: 8, lower bound: -1.4966896, upper bound: 1.4969374
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.39
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

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5788

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5832

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4924341, upper bound: 1.4969311
time: 9.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966854, upper bound: 1.4926797
time: 9.97 seconds

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

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5788

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4772432, upper bound: 1.4966758
time: 6.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4969218, upper bound: 1.4769974
time: 6.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 26.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.09
Output dim: 8, lower bound: -1.4924341, upper bound: 1.4969311
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.09
Output dim: 8, lower bound: -1.4966854, upper bound: 1.4926797
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 26.09
Output dim: 8, lower bound: -1.4772432, upper bound: 1.4966758
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 26.09
Output dim: 8, lower bound: -1.4969218, upper bound: 1.4769974

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5833745, 2.5806825
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7304134, 2.7228594
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.4043169, 2.4050732
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7417355, 2.7513652
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5625825, 1.5666995
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3523335, 2.3543880
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4309597, 2.4337707
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9705172, 2.9627075

Time for backsubstitution: 12.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 822

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4671

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4887168, upper bound: 1.4969254
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4924285, upper bound: 1.4932116
time: 6.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5839944, 2.5800626
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7248955, 2.7283773
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.4063578, 2.4030313
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7470846, 2.7460170
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5635867, 1.5656958
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3485217, 2.3581994
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4337521, 2.4309783
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9657297, 2.9674954

Time for backsubstitution: 12.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 6109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4917957, upper bound: 1.4926715
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966773, upper bound: 1.4877904
time: 6.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5581226, 2.5665851
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6685276, 2.6437521
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.4073172, 2.4077253
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7644825, 2.7669513
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5384045, 1.5254433
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2954426, 2.2665811
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3756862, 2.3930326
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9807715, 2.9866476

Time for backsubstitution: 12.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4772327, upper bound: 1.4908812
time: 6.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4714487, upper bound: 1.4966655
time: 6.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5648022, 2.5599053
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6426620, 2.6696172
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.4070330, 2.4080095
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7692547, 2.7621794
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5271196, 1.5367281
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2697353, 2.2922878
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3930430, 2.3756747
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9850173, 2.9824028

Time for backsubstitution: 12.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6124

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4942279, upper bound: 1.4769959
time: 7.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4969206, upper bound: 1.4743053
time: 4.99 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.79 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.79
Output dim: 8, lower bound: -1.4887168, upper bound: 1.4969254
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.79
Output dim: 8, lower bound: -1.4924285, upper bound: 1.4932116
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.79
Output dim: 8, lower bound: -1.4917957, upper bound: 1.4926715
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.79
Output dim: 8, lower bound: -1.4966773, upper bound: 1.4877904
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.79
Output dim: 8, lower bound: -1.4772327, upper bound: 1.4908812
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.79
Output dim: 8, lower bound: -1.4714487, upper bound: 1.4966655
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.79
Output dim: 8, lower bound: -1.4942279, upper bound: 1.4769959
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.79
Output dim: 8, lower bound: -1.4969206, upper bound: 1.4743053

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5785680, 2.5661044
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7224741, 2.6986976
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3955173, 2.3784060
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7393718, 2.7442207
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5529256, 1.5635276
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3483472, 2.3422866
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4235516, 2.4313326
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9702578, 2.9626164

Time for backsubstitution: 12.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5788

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6124

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4860199, upper bound: 1.4969241
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4887156, upper bound: 1.4942318
time: 6.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5687966, 2.5758755
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7062521, 2.7149181
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3776493, 2.3962731
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7345901, 2.7489986
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5594106, 1.5570425
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3402314, 2.3504002
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4285202, 2.4263630
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9704227, 2.9624486

Time for backsubstitution: 12.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 822

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4924180, upper bound: 1.4874180
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4866340, upper bound: 1.4932010
time: 6.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6186500, 2.6057091
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6997743, 2.6859608
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3687162, 2.3763590
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7033165, 2.6842463
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5549526, 1.5535159
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3348389, 2.3388865
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4194221, 2.4238973
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9517131, 2.9477201

Time for backsubstitution: 12.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5788

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6124

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4891018, upper bound: 1.4926704
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4917945, upper bound: 1.4899776
time: 5.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6096416, 2.6147189
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6824803, 2.7032557
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3796864, 2.3653898
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6853139, 2.7022491
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5514059, 1.5570621
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3292093, 2.3445158
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4266710, 2.4166484
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9459529, 2.9534788

Time for backsubstitution: 12.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 4671

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6127

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4911472, upper bound: 1.4877797
time: 6.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4966666, upper bound: 1.4822597
time: 6.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5576291, 2.5648930
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6676502, 2.6434903
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.4061813, 2.4038281
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7455215, 2.7614377
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5352569, 1.5145042
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2947145, 2.2640765
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3727837, 2.3829684
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9672809, 2.9827662

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6127

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4717025, upper bound: 1.4908704
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4772219, upper bound: 1.4853538
time: 5.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5564313, 2.5660911
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6682644, 2.6428761
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.4034195, 2.4065895
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7589674, 2.7479920
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5274653, 1.5222952
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2929378, 2.2658532
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3656216, 2.3901300
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9768901, 2.9731560

Time for backsubstitution: 12.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 6124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4665590, upper bound: 1.4966573
time: 5.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4714406, upper bound: 1.4917757
time: 6.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5533400, 2.5558960
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6400709, 2.6621943
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3960743, 2.4041915
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7668231, 2.7551982
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5243940, 1.5289006
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2676497, 2.2862928
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3900542, 2.3746285
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9847298, 2.9815893

Time for backsubstitution: 12.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5832

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4899725, upper bound: 1.4769921
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4942237, upper bound: 1.4727403
time: 6.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5607929, 2.5484428
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6352386, 2.6670256
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.4032145, 2.3970513
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7622740, 2.7597473
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5192928, 1.5340018
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2637405, 2.2902014
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3919978, 2.3726845
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9842033, 2.9821167

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5788

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4671

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4932013, upper bound: 1.4742977
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4969150, upper bound: 1.4705821
time: 6.22 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 25.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 8, lower bound: -1.4860199, upper bound: 1.4969241
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 8, lower bound: -1.4887156, upper bound: 1.4942318
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 8, lower bound: -1.4924180, upper bound: 1.4874180
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 8, lower bound: -1.4866340, upper bound: 1.4932010
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 8, lower bound: -1.4891018, upper bound: 1.4926704
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 8, lower bound: -1.4917945, upper bound: 1.4899776
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 8, lower bound: -1.4911472, upper bound: 1.4877797
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 8, lower bound: -1.4966666, upper bound: 1.4822597
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 8, lower bound: -1.4717025, upper bound: 1.4908704
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 8, lower bound: -1.4772219, upper bound: 1.4853538
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 8, lower bound: -1.4665590, upper bound: 1.4966573
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 8, lower bound: -1.4714406, upper bound: 1.4917757
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 8, lower bound: -1.4899725, upper bound: 1.4769921
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 8, lower bound: -1.4942237, upper bound: 1.4727403
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 8, lower bound: -1.4932013, upper bound: 1.4742977
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.71
Output dim: 8, lower bound: -1.4969150, upper bound: 1.4705821

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5671048, 2.5620949
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7198820, 2.6912742
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3845587, 2.3745866
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7369390, 2.7372398
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5502000, 1.5557007
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3462605, 2.3362916
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4205627, 2.4302859
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9699721, 2.9618034

Time for backsubstitution: 12.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4860019, upper bound: 1.4963383
time: 7.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4854330, upper bound: 1.4969062
time: 7.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5745578, 2.5546417
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7150507, 2.6961055
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3916979, 2.3674469
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2412071, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7323909, 2.7417886
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5450988, 1.5608019
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3423524, 2.3401999
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4225063, 2.4283419
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9694448, 2.9623308

Time for backsubstitution: 12.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 822

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4719776, upper bound: 1.4937795
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4882614, upper bound: 1.4774935
time: 5.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5683031, 2.5741842
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7053747, 2.7146559
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3765144, 2.3923769
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7156315, 2.7434850
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5562625, 1.5461036
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3395033, 2.3478954
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4256182, 2.4162998
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9569302, 2.9585667

Time for backsubstitution: 12.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4875283, upper bound: 1.4874118
time: 7.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4924099, upper bound: 1.4825283
time: 6.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5671053, 2.5753822
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.7059889, 2.7140417
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3737526, 2.3951383
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7290764, 2.7300394
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5484719, 1.5538948
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3377266, 2.3496721
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4184561, 2.4234614
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9665403, 2.9489565

Time for backsubstitution: 12.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 822

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4698959, upper bound: 1.4927464
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4861798, upper bound: 1.4764629
time: 6.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6071868, 2.6016994
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6971831, 2.6785374
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3577576, 2.3725400
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2403488, 3.2383361
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7008867, 2.6772673
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5522292, 1.5456910
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3327518, 2.3328917
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4164324, 2.4228511
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9514275, 2.9469085

Time for backsubstitution: 12.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 822

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4890838, upper bound: 1.4920862
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4885153, upper bound: 1.4926547
time: 6.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6146388, 2.5942459
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6923518, 2.6833687
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3648977, 2.3654008
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2304678, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6963377, 2.6818161
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5471280, 1.5507922
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3288436, 2.3368003
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4183760, 2.4209075
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9509010, 2.9474354

Time for backsubstitution: 12.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 822

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5736

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4721022, upper bound: 1.4899643
time: 6.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4917809, upper bound: 1.4702859
time: 6.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5878453, 2.6072731
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6769724, 2.6872349
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3556385, 2.3571444
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6796818, 2.6859050
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5400119, 1.5237927
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.3133259, 2.2980990
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4187946, 2.4139385
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9367118, 2.9503117

Time for backsubstitution: 12.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4911366, upper bound: 1.4819848
time: 9.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4853527, upper bound: 1.4877690
time: 6.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.6021953, 2.5929229
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6664591, 2.6977482
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3714409, 2.3413420
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2407608
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6689701, 2.6966169
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5181365, 1.5456676
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2827926, 2.3286324
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.4239616, 2.4087720
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9427867, 2.9442368

Time for backsubstitution: 12.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 5788

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 918

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4921568, upper bound: 1.4819763
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4959939, upper bound: 1.4819903
time: 6.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5358262, 2.5574358
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6621280, 2.6274600
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3821344, 2.3955836
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7398908, 2.7450943
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5238328, 1.4812245
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2787690, 2.2176378
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3649063, 2.3802581
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9580379, 2.9795942

Time for backsubstitution: 12.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 918

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6137

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4668129, upper bound: 1.4908621
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4716945, upper bound: 1.4859804
time: 5.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5501771, 2.5430903
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6516204, 2.6379728
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3979340, 2.3797812
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7291782, 2.7558115
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5019770, 1.5030994
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2482753, 2.2481711
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3700724, 2.3750911
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9641128, 2.9735236

Time for backsubstitution: 12.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 4671

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 822

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4762849, upper bound: 1.4848992
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4762859, upper bound: 1.4648391
time: 6.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5910878, 2.5917382
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6431437, 2.6004610
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3657799, 2.3799191
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7151997, 2.6862214
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5188322, 1.5101156
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2792544, 2.2465410
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3512917, 2.3830490
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9628744, 2.9533815

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 822

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4656221, upper bound: 1.4962045
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4656228, upper bound: 1.4761456
time: 5.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5820785, 2.6007478
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6258488, 2.6177559
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3767500, 2.3689494
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2429070
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6971972, 2.7042241
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5152855, 1.5136621
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2736249, 2.2521703
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3585405, 2.3758006
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9571161, 2.9591403

Time for backsubstitution: 12.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4671

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4677225, upper bound: 1.4917703
time: 6.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4714350, upper bound: 1.4880573
time: 7.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5523863, 2.5555620
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6372037, 2.6538105
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3929663, 2.4031248
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7586806, 2.7524064
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5228634, 1.5283742
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2656617, 2.2804945
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3858047, 2.3731723
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9822407, 2.9743118

Time for backsubstitution: 12.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 4671

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 918

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4897024, upper bound: 1.4763191
time: 6.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4896886, upper bound: 1.4724840
time: 5.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5530062, 2.5549421
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6316857, 2.6593285
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3950071, 2.4010830
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7640297, 2.7470567
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5238667, 1.5273705
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2618508, 2.2843058
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3885970, 2.3703799
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9774532, 2.9790998

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 822

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6109

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4942133, upper bound: 1.4669460
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4884294, upper bound: 1.4727300
time: 6.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5559864, 2.5338645
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6272979, 2.6428633
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3944149, 2.3703837
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2435040, 3.2435040
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.7599099, 2.7526023
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5096354, 1.5308301
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2597513, 2.2780983
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3845897, 2.3702440
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.9839439, 2.9820251

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5832

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6127

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4876716, upper bound: 1.4742871
time: 6.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4931901, upper bound: 1.4687674
time: 6.78 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.85 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4860019, upper bound: 1.4963383
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4854330, upper bound: 1.4969062
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4719776, upper bound: 1.4937795
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4882614, upper bound: 1.4774935
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4875283, upper bound: 1.4874118
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4924099, upper bound: 1.4825283
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4698959, upper bound: 1.4927464
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4861798, upper bound: 1.4764629
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4890838, upper bound: 1.4920862
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4885153, upper bound: 1.4926547
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4721022, upper bound: 1.4899643
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4917809, upper bound: 1.4702859
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4911366, upper bound: 1.4819848
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4853527, upper bound: 1.4877690
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4921568, upper bound: 1.4819763
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4959939, upper bound: 1.4819903
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4668129, upper bound: 1.4908621
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4716945, upper bound: 1.4859804
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4762849, upper bound: 1.4848992
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4762859, upper bound: 1.4648391
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4656221, upper bound: 1.4962045
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4656228, upper bound: 1.4761456
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4677225, upper bound: 1.4917703
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4714350, upper bound: 1.4880573
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4897024, upper bound: 1.4763191
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4896886, upper bound: 1.4724840
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4942133, upper bound: 1.4669460
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4884294, upper bound: 1.4727300
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4876716, upper bound: 1.4742871
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.85
Output dim: 8, lower bound: -1.4931901, upper bound: 1.4687674
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 25.85
Output dim: 8, lower bound: -1.4969150, upper bound: 1.4705821
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=2.435220718383789
rel_dist={8: [-1.4969366874007388, 1.496938647628781]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5832
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5832

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3809741, upper bound: 1.3847671
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847651, upper bound: 1.3809744
time: 4.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 11.04 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 11.04
Output dim: 8, lower bound: -1.3809741, upper bound: 1.3847671
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 11.04
Output dim: 8, lower bound: -1.3847651, upper bound: 1.3809744

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4961901, 2.4967217
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6521859, 2.6474566
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3210001, 2.3227506
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2015905, 3.2019906
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6222701, 2.6268556
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5011230, 1.5019834
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2621202, 2.2588534
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3552809, 2.3576746
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8893456, 2.8852425

Time for backsubstitution: 12.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 822

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4556

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3779157, upper bound: 1.3847516
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3809604, upper bound: 1.3817069
time: 4.22 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4967222, 2.4961905
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6474566, 2.6521859
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3227501, 2.3210006
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.2019911, 3.2015905
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6268554, 2.6222703
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.5019836, 1.5011232
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2588534, 2.2621202
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3576746, 2.3552809
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8852420, 2.8893461

Time for backsubstitution: 12.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4556
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 4671

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4556

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3817068, upper bound: 1.3809602
time: 6.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847513, upper bound: 1.3779161
time: 4.32 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.30 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.30
Output dim: 8, lower bound: -1.3779157, upper bound: 1.3847516
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 23.30
Output dim: 8, lower bound: -1.3809604, upper bound: 1.3817069
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 23.30
Output dim: 8, lower bound: -1.3817068, upper bound: 1.3809602
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.30
Output dim: 8, lower bound: -1.3847513, upper bound: 1.3779161

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4928746, 2.4966397
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6561866, 2.6470847
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.2983208, 2.3093915
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1878605, 3.1836796
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6079636, 2.5996399
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4809020, 1.4744284
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2474580, 2.2373309
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3477945, 2.3520613
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8758841, 2.8751421

Time for backsubstitution: 12.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 4671

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6127

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3731570, upper bound: 1.3847424
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3779067, upper bound: 1.3799945
time: 5.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4966407, 2.4928746
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6470847, 2.6561871
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3093915, 2.2983208
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1836796, 3.1878605
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.5996399, 2.6079638
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4744284, 1.4809022
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2373309, 2.2474582
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3520613, 2.3477945
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8751421, 2.8758836

Time for backsubstitution: 12.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 6127

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5788

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.3847359, upper bound: 1.3773275
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3841610, upper bound: 1.3779005
time: 4.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.87 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.87
Output dim: 8, lower bound: -1.3731570, upper bound: 1.3847424
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 21.87
Output dim: 8, lower bound: -1.3779067, upper bound: 1.3799945
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.87
Output dim: 8, lower bound: -1.3847359, upper bound: 1.3773275
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 21.87
Output dim: 8, lower bound: -1.3841610, upper bound: 1.3779005

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.4710784, 2.4871435
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6491780, 2.6310644
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.2742720, 2.2988877
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1821585, 3.1811919
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6008010, 2.5832956
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4663830, 1.4411592
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2272129, 2.1909142
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3399181, 2.3486133
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8666410, 2.8711066

Time for backsubstitution: 12.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 5788
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 6137
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6124

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 822

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3592464, upper bound: 1.3842541
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3726669, upper bound: 1.3708325
time: 6.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.4408283, -5.7609482, -8.4408283, -5.7609482, -2.5010200, 2.4893148
1: -10.8713818, -7.8377485, -10.8713818, -7.8377485, -2.6483693, 2.6551418
2: -5.0804310, -2.3590326, -5.0804310, -2.3590326, -2.3063593, 2.3020492
3: -6.1261268, -2.8826227, -6.1261268, -2.8826227, -3.1842661, 3.1873894
4: -13.4648161, -9.8270741, -13.4648161, -9.8270741, -2.6052618, 2.6033683
5: -3.5627654, -1.4808009, -3.5627654, -1.4808009, -1.4781189, 1.4779031
6: -10.8806124, -8.0272884, -10.8806124, -8.0272884, -2.2370529, 2.2477970
7: -9.6614361, -6.2756395, -9.6614361, -6.2756395, -3.3857965, 3.3857965
8: 9.2794437, 11.9540062, 9.2794437, 11.9540062, -2.3509150, 2.3492279
9: -7.8706493, -4.4301844, -7.8706493, -4.4301844, -2.8805733, 2.8714638

Time for backsubstitution: 12.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 6124
type: RSZ, layer: 1, pos: 5830
type: RSZ, layer: 1, pos: 6127
type: RSZ, layer: 1, pos: 4671
type: RSZ, layer: 1, pos: 918
type: RSZ, layer: 1, pos: 6109
type: RSZ, layer: 1, pos: 5736
type: RSZ, layer: 1, pos: 6137

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 822

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3708275, upper bound: 1.3768372
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -1.3842456, upper bound: 1.3634167
time: 4.13 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.16 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 21.16
Output dim: 8, lower bound: -1.3592464, upper bound: 1.3842541
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 21.16
Output dim: 8, lower bound: -1.3726669, upper bound: 1.3708325
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 21.16
Output dim: 8, lower bound: -1.3708275, upper bound: 1.3768372
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 21.16
Output dim: 8, lower bound: -1.3842456, upper bound: 1.3634167
Binary search (step 2): status=Status.VERIFIED, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=2.359529972076416
rel_dist={8: [-1.3847687820365167, 1.3847690605870273]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 1810.19 seconds
