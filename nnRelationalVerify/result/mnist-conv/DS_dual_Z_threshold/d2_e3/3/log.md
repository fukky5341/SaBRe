## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.23181255539999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9982100, 0.9982100)
1: (-10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.9129472, 0.9129472)
2: (-8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5460193, 0.5460193)
3: (-3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6636717, 0.6636720)
4: (-10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1278839, 1.1278849)
5: (8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6244779, 0.6244783)
6: (-7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6638887, 0.6638887)
7: (-12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1221027, 1.1221032)
8: (-1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5523930, 0.5523930)
9: (-3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.8008251, 0.8008251)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.22 + 33.12 = 56.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.2320443, upper bound: 0.2320458

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 5848
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 511

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2320439, upper bound: 0.2317790
time: 4.52 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2317775, upper bound: 0.2320454
time: 3.56 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.37 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.37
Output dim: 5, lower bound: -0.2320439, upper bound: 0.2317790
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.37
Output dim: 5, lower bound: -0.2317775, upper bound: 0.2320454

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9934788, 0.9925323
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.9097400, 0.9090981
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5389168, 0.5374341
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6598189, 0.6590480
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1199093, 1.1212382
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6244764, 0.6244760
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6611404, 0.6605909
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1220989, 1.1221051
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5397756, 0.5372114
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7904727, 0.7922003

Time for backsubstitution: 20.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 5848
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2318110, upper bound: 0.2317785
time: 4.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2320434, upper bound: 0.2315450
time: 4.80 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9925323, 0.9934788
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.9090977, 0.9097400
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5374341, 0.5389168
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6590481, 0.6598189
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1212387, 1.1199088
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6244760, 0.6244764
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6605911, 0.6611407
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1221056, 1.1220989
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5372114, 0.5397756
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7922003, 0.7904730

Time for backsubstitution: 21.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 5848
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2315446, upper bound: 0.2320449
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2317770, upper bound: 0.2318113
time: 4.27 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.04 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 30.04
Output dim: 5, lower bound: -0.2318110, upper bound: 0.2317785
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.04
Output dim: 5, lower bound: -0.2320434, upper bound: 0.2315450
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.04
Output dim: 5, lower bound: -0.2315446, upper bound: 0.2320449
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 30.04
Output dim: 5, lower bound: -0.2317770, upper bound: 0.2318113

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9893613, 0.9849415
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.9062662, 0.9026947
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5380573, 0.5369663
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6558516, 0.6517400
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1166172, 1.1194520
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6238251, 0.6241264
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6594315, 0.6574447
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1198921, 1.1180339
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5387492, 0.5366552
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7883685, 0.7910585

Time for backsubstitution: 21.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5848
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 5848

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2320433, upper bound: 0.2315445
time: 4.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2320433, upper bound: 0.2315445
time: 4.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9849415, 0.9893613
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.9026947, 0.9062662
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5369663, 0.5380573
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6517398, 0.6558515
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1194515, 1.1166172
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6241264, 0.6238251
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6574450, 0.6594312
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1180334, 1.1198921
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5366552, 0.5387492
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7910583, 0.7883682

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5848
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 5848

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2315445, upper bound: 0.2320448
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2315445, upper bound: 0.2320436
time: 4.64 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.64 seconds
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 5, lower bound: -0.2320433, upper bound: 0.2315445
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 5, lower bound: -0.2320433, upper bound: 0.2315445
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 5, lower bound: -0.2315445, upper bound: 0.2320448
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 5, lower bound: -0.2315445, upper bound: 0.2320436

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9888468, 0.9846916
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.9026923, 0.8953371
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5380218, 0.5368941
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6555388, 0.6510977
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1140776, 1.1182184
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6237597, 0.6239910
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6579621, 0.6544170
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1194496, 1.1171250
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5363991, 0.5318203
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7849920, 0.7894194

Time for backsubstitution: 21.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 5861

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2319455, upper bound: 0.2315454
time: 3.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2320427, upper bound: 0.2314473
time: 4.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9891114, 0.9844270
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8989086, 0.8991208
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5379851, 0.5369308
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6552091, 0.6514274
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1153841, 1.1169114
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6236901, 0.6240606
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6564038, 0.6559758
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1189823, 1.1175919
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5339143, 0.5343051
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7867296, 0.7876821

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 5861

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2319455, upper bound: 0.2315454
time: 3.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2320427, upper bound: 0.2314473
time: 4.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9844270, 0.9891114
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8991208, 0.8989086
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5369308, 0.5379851
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6514273, 0.6552092
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1169119, 1.1153836
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6240611, 0.6236906
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6559756, 0.6564035
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1175919, 1.1189833
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5343051, 0.5339143
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7876823, 0.7867293

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 5861

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2314467, upper bound: 0.2320432
time: 4.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2315440, upper bound: 0.2319470
time: 3.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9846916, 0.9888468
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8953371, 0.9026923
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5368941, 0.5380218
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6510975, 0.6555389
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1182184, 1.1140771
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6239915, 0.6237593
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6544173, 0.6579623
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1171246, 1.1194496
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5318203, 0.5363991
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7894194, 0.7849920

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 5861

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2314467, upper bound: 0.2320432
time: 5.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2315440, upper bound: 0.2319470
time: 3.32 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.61 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 5, lower bound: -0.2319455, upper bound: 0.2315454
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 5, lower bound: -0.2320427, upper bound: 0.2314473
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 5, lower bound: -0.2319455, upper bound: 0.2315454
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 5, lower bound: -0.2320427, upper bound: 0.2314473
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 5, lower bound: -0.2314467, upper bound: 0.2320432
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 5, lower bound: -0.2315440, upper bound: 0.2319470
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 5, lower bound: -0.2314467, upper bound: 0.2320432
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 5, lower bound: -0.2315440, upper bound: 0.2319470

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9883189, 0.9843407
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.9011731, 0.8930535
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5379517, 0.5367882
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6546686, 0.6505189
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1140766, 1.1182179
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6236920, 0.6238909
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6576965, 0.6540196
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1162796, 1.1150203
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5358496, 0.5314548
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7845898, 0.7888150

Time for backsubstitution: 21.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 831

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2317095, upper bound: 0.2302635
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2306625, upper bound: 0.2313106
time: 3.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9884963, 0.9841638
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.9004087, 0.8938203
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5379159, 0.5368240
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6549602, 0.6502273
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1140776, 1.1182179
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6236591, 0.6239243
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6575644, 0.6541526
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1173439, 1.1139560
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5360336, 0.5312705
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7843876, 0.7890174

Time for backsubstitution: 22.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 831

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2318076, upper bound: 0.2301654
time: 4.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2307606, upper bound: 0.2312125
time: 3.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9885836, 0.9840760
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8973918, 0.8968372
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5379150, 0.5368249
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6543388, 0.6508486
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1153841, 1.1169114
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6236229, 0.6239600
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6561391, 0.6555784
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1158142, 1.1154866
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5333648, 0.5339396
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7863274, 0.7870777

Time for backsubstitution: 22.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 831

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2317095, upper bound: 0.2302635
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2306625, upper bound: 0.2313106
time: 3.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9887605, 0.9838991
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8966250, 0.8976016
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5378792, 0.5368607
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6546304, 0.6505570
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1153841, 1.1169114
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6235895, 0.6239934
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6560061, 0.6557100
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1168785, 1.1144223
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5335488, 0.5337555
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7861247, 0.7872798

Time for backsubstitution: 21.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 831

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2318076, upper bound: 0.2301642
time: 4.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2307606, upper bound: 0.2312125
time: 3.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9838991, 0.9887605
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8976016, 0.8966250
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5368605, 0.5378792
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6505570, 0.6546304
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1169119, 1.1153836
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6239934, 0.6235895
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6557105, 0.6560059
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1144218, 1.1168780
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5337553, 0.5335488
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7872801, 0.7861247

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 831

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2312110, upper bound: 0.2307620
time: 4.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2301640, upper bound: 0.2318081
time: 4.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9840760, 0.9885836
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8968372, 0.8973918
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5368247, 0.5379150
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6508486, 0.6543388
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1169119, 1.1153836
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6239600, 0.6236229
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6555784, 0.6561391
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1154861, 1.1158137
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5339396, 0.5333648
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7870779, 0.7863271

Time for backsubstitution: 21.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 831

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2313091, upper bound: 0.2306639
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2302621, upper bound: 0.2317110
time: 3.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9841638, 0.9884958
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8938203, 0.9004087
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5368240, 0.5379159
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6502273, 0.6549602
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1182184, 1.1140771
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6239243, 0.6236591
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6541526, 0.6575646
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1139565, 1.1173444
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5312705, 0.5360336
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7890177, 0.7843874

Time for backsubstitution: 21.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 831

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2312110, upper bound: 0.2307620
time: 4.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2301640, upper bound: 0.2318081
time: 4.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9843407, 0.9883189
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8930535, 0.9011731
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5367882, 0.5379517
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6505189, 0.6546686
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1182184, 1.1140766
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6238909, 0.6236920
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6540196, 0.6576965
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1150208, 1.1162801
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5314546, 0.5358496
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7888150, 0.7845898

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 831

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2313091, upper bound: 0.2306639
time: 4.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2302621, upper bound: 0.2317110
time: 4.11 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 30.76 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.76
Output dim: 5, lower bound: -0.2317095, upper bound: 0.2302635
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.76
Output dim: 5, lower bound: -0.2306625, upper bound: 0.2313106
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.76
Output dim: 5, lower bound: -0.2318076, upper bound: 0.2301654
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.76
Output dim: 5, lower bound: -0.2307606, upper bound: 0.2312125
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.76
Output dim: 5, lower bound: -0.2317095, upper bound: 0.2302635
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.76
Output dim: 5, lower bound: -0.2306625, upper bound: 0.2313106
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.76
Output dim: 5, lower bound: -0.2318076, upper bound: 0.2301642
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.76
Output dim: 5, lower bound: -0.2307606, upper bound: 0.2312125
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.76
Output dim: 5, lower bound: -0.2312110, upper bound: 0.2307620
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.76
Output dim: 5, lower bound: -0.2301640, upper bound: 0.2318081
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.76
Output dim: 5, lower bound: -0.2313091, upper bound: 0.2306639
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.76
Output dim: 5, lower bound: -0.2302621, upper bound: 0.2317110
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.76
Output dim: 5, lower bound: -0.2312110, upper bound: 0.2307620
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.76
Output dim: 5, lower bound: -0.2301640, upper bound: 0.2318081
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.76
Output dim: 5, lower bound: -0.2313091, upper bound: 0.2306639
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.76
Output dim: 5, lower bound: -0.2302621, upper bound: 0.2317110

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 56.34 + 496.86 = 553.20 seconds
