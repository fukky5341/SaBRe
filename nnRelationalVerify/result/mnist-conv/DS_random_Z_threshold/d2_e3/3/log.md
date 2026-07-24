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
execution time: IAR + RelationalAnalysis = 24.06 + 32.77 = 56.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.2320443, upper bound: 0.2320458

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5848
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5848

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2320441, upper bound: 0.2320457
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2320441, upper bound: 0.2320456
time: 3.15 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.26 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.26
Output dim: 5, lower bound: -0.2320441, upper bound: 0.2320457
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.26
Output dim: 5, lower bound: -0.2320441, upper bound: 0.2320456

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9976954, 0.9979601
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.9093742, 0.9055905
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5459843, 0.5459476
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6633589, 0.6630293
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1253443, 1.1266513
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6244121, 0.6243429
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6624196, 0.6608605
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1216617, 1.1211948
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5500422, 0.5475574
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7974486, 0.7991860

Time for backsubstitution: 23.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 5861

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6139

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2320404, upper bound: 0.2297884
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2297868, upper bound: 0.2320419
time: 3.90 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9979596, 0.9976954
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.9055905, 0.9093742
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5459476, 0.5459843
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6630292, 0.6633589
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1266518, 1.1253443
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6243429, 0.6244121
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6608608, 0.6624193
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1211944, 1.1216612
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5475571, 0.5500422
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7991858, 0.7974486

Time for backsubstitution: 23.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6139
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 831

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6139

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2320404, upper bound: 0.2297884
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2297868, upper bound: 0.2320419
time: 3.64 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.10 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.10
Output dim: 5, lower bound: -0.2320404, upper bound: 0.2297884
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.10
Output dim: 5, lower bound: -0.2297868, upper bound: 0.2320419
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.10
Output dim: 5, lower bound: -0.2320404, upper bound: 0.2297884
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.10
Output dim: 5, lower bound: -0.2297868, upper bound: 0.2320419

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9962692, 0.9967675
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8998842, 0.8983555
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5451260, 0.5457387
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6537461, 0.6552093
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1151752, 1.1181722
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6140494, 0.6113925
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6556964, 0.6552560
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1025167, 1.1052332
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5508943, 0.5489657
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7981911, 0.7997599

Time for backsubstitution: 22.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 831

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2318790, upper bound: 0.2294389
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2316909, upper bound: 0.2296270
time: 3.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9965029, 0.9965339
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.9021392, 0.8961005
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5457752, 0.5450892
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6555390, 0.6534164
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1168652, 1.1164813
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6114621, 0.6139798
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6568151, 0.6541374
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1056995, 1.1020503
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5514505, 0.5484095
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7980223, 0.7999282

Time for backsubstitution: 22.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 831

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 511

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2297865, upper bound: 0.2317751
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2295201, upper bound: 0.2320415
time: 3.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9965339, 0.9965034
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8961005, 0.9021392
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5450892, 0.5457752
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6534164, 0.6555390
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1164818, 1.1168656
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6139798, 0.6114621
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6541371, 0.6568148
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1020503, 1.1056995
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5484095, 0.5514505
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7999282, 0.7980225

Time for backsubstitution: 23.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 831

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 947

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2319301, upper bound: 0.2296738
time: 4.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2319258, upper bound: 0.2296782
time: 3.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9967675, 0.9962692
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8983555, 0.8998842
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5457387, 0.5451260
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6552093, 0.6537461
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1181726, 1.1151748
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6113925, 0.6140494
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6552563, 0.6556962
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1052332, 1.1025167
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5489657, 0.5508943
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7997599, 0.7981908

Time for backsubstitution: 23.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2296255, upper bound: 0.2316924
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2294374, upper bound: 0.2318805
time: 3.58 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.24 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.24
Output dim: 5, lower bound: -0.2318790, upper bound: 0.2294389
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 30.24
Output dim: 5, lower bound: -0.2316909, upper bound: 0.2296270
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 30.24
Output dim: 5, lower bound: -0.2297865, upper bound: 0.2317751
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.24
Output dim: 5, lower bound: -0.2295201, upper bound: 0.2320415
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.24
Output dim: 5, lower bound: -0.2319301, upper bound: 0.2296738
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.24
Output dim: 5, lower bound: -0.2319258, upper bound: 0.2296782
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 30.24
Output dim: 5, lower bound: -0.2296255, upper bound: 0.2316924
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.24
Output dim: 5, lower bound: -0.2294374, upper bound: 0.2318805

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9962692, 0.9967675
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8998842, 0.8983550
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5451260, 0.5457387
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6537459, 0.6552092
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1151733, 1.1181707
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6140485, 0.6113915
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6556938, 0.6552529
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1025171, 1.1052327
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5508921, 0.5489626
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7981887, 0.7997587

Time for backsubstitution: 22.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 511

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5861

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2317812, upper bound: 0.2294384
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2318785, upper bound: 0.2293411
time: 4.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9908252, 0.9918027
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8982902, 0.8928938
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5371890, 0.5379860
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6509149, 0.6495631
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1102185, 1.1085052
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6114597, 0.6139779
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6535172, 0.6513891
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1057010, 1.1020455
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5362692, 0.5357924
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7893975, 0.7895758

Time for backsubstitution: 23.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 5861

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4667

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2292788, upper bound: 0.2320380
time: 3.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2295165, upper bound: 0.2318003
time: 4.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9965291, 0.9965162
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8962359, 0.9020953
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5450504, 0.5458925
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6534407, 0.6555310
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1164761, 1.1168838
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6140699, 0.6114321
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6541348, 0.6568222
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1020117, 1.1058197
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5483954, 0.5514925
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7999163, 0.7980614

Time for backsubstitution: 23.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 4667

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 511

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2319298, upper bound: 0.2294071
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2316634, upper bound: 0.2296735
time: 3.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9965339, 0.9964986
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8960567, 0.9021392
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5450892, 0.5457363
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6534085, 0.6555390
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1164818, 1.1168594
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6139503, 0.6114621
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6541371, 0.6568122
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1020503, 1.1056609
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5484095, 0.5514364
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7999282, 0.7980103

Time for backsubstitution: 22.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 511

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2317645, upper bound: 0.2293287
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2315764, upper bound: 0.2295156
time: 4.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9967675, 0.9962692
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8983550, 0.8998842
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5457387, 0.5451260
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6552093, 0.6537460
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1181707, 1.1151733
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6113915, 0.6140485
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6552527, 0.6556938
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1052327, 1.1025171
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5489626, 0.5508921
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7997589, 0.7981887

Time for backsubstitution: 23.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 5861

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 947

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2293271, upper bound: 0.2317660
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2293229, upper bound: 0.2317703
time: 3.59 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.89 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.89
Output dim: 5, lower bound: -0.2317812, upper bound: 0.2294384
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.89
Output dim: 5, lower bound: -0.2318785, upper bound: 0.2293411
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.89
Output dim: 5, lower bound: -0.2292788, upper bound: 0.2320380
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.89
Output dim: 5, lower bound: -0.2295165, upper bound: 0.2318003
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.89
Output dim: 5, lower bound: -0.2319298, upper bound: 0.2294071
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.89
Output dim: 5, lower bound: -0.2316634, upper bound: 0.2296735
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.89
Output dim: 5, lower bound: -0.2317645, upper bound: 0.2293287
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.89
Output dim: 5, lower bound: -0.2315764, upper bound: 0.2295156
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.89
Output dim: 5, lower bound: -0.2293271, upper bound: 0.2317660
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.89
Output dim: 5, lower bound: -0.2293229, upper bound: 0.2317703

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9959183, 0.9962392
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8976007, 0.8968377
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5450196, 0.5456681
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6531668, 0.6543384
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1151733, 1.1181703
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6139479, 0.6113238
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6552968, 0.6549888
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1004124, 1.1020637
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5505264, 0.5484126
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7975841, 0.7993562

Time for backsubstitution: 23.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 511

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 831

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2316443, upper bound: 0.2280597
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2305973, upper bound: 0.2291067
time: 4.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9919796, 0.9915094
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.9001622, 0.8924227
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5368848, 0.5391877
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6508644, 0.6497819
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1109724, 1.1083155
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6119633, 0.6138573
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6554708, 0.6508970
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1051211, 1.1043177
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5368974, 0.5356324
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7893765, 0.7896597

Time for backsubstitution: 23.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 831

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2291174, upper bound: 0.2316082
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2291155, upper bound: 0.2318766
time: 5.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9917979, 0.9908381
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8930287, 0.8982458
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5379474, 0.5373068
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6495874, 0.6509069
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1084991, 1.1102366
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6140680, 0.6114306
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6513867, 0.6535244
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1020060, 1.1058207
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5357785, 0.5363114
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7895639, 0.7894366

Time for backsubstitution: 22.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4667
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4667

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2316885, upper bound: 0.2294035
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2319262, upper bound: 0.2291659
time: 3.56 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 29.20 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 29.20
Output dim: 5, lower bound: -0.2316443, upper bound: 0.2280597
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.20
Output dim: 5, lower bound: -0.2305973, upper bound: 0.2291067
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 29.20
Output dim: 5, lower bound: -0.2291174, upper bound: 0.2316082
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 29.20
Output dim: 5, lower bound: -0.2291155, upper bound: 0.2318766
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 29.20
Output dim: 5, lower bound: -0.2316885, upper bound: 0.2294035
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 29.20
Output dim: 5, lower bound: -0.2319262, upper bound: 0.2291659

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9919801, 0.9915094
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.9001603, 0.8924217
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5368848, 0.5391874
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6508644, 0.6497818
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1109705, 1.1083140
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6119623, 0.6138568
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6554675, 0.6508949
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1051216, 1.1043181
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5368974, 0.5356333
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7893746, 0.7896571

Time for backsubstitution: 22.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 5861

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 831

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2288833, upper bound: 0.2305953
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2278370, upper bound: 0.2316424
time: 3.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9915042, 0.9908381
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8925576, 0.8982458
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5379474, 0.5370026
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6495874, 0.6508566
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1083097, 1.1102366
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6139479, 0.6114306
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6508951, 0.6535244
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.1020060, 1.1052408
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5356185, 0.5363114
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7895639, 0.7894146

Time for backsubstitution: 22.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5861
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5861

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2319122, upper bound: 0.2291562
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.2319120, upper bound: 0.2291562
time: 3.30 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 29.45 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 29.45
Output dim: 5, lower bound: -0.2288833, upper bound: 0.2305953
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 29.45
Output dim: 5, lower bound: -0.2278370, upper bound: 0.2316424
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 29.45
Output dim: 5, lower bound: -0.2319122, upper bound: 0.2291562
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 29.45
Output dim: 5, lower bound: -0.2319120, upper bound: 0.2291562

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9909763, 0.9904876
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8910398, 0.8959618
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5378773, 0.5368969
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6487164, 0.6502775
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1083088, 1.1102362
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6138806, 0.6113300
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6506300, 0.6531267
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.0988369, 1.1031361
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5350688, 0.5359457
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7891614, 0.7888098

Time for backsubstitution: 22.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2317508, upper bound: 0.2289929
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.2314824, upper bound: 0.2289947
time: 3.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.7666721, -9.2695217, -10.7666721, -9.2695217, -0.9911537, 0.9903102
1: -10.7616920, -9.5120468, -10.7616920, -9.5120468, -0.8902736, 0.8967261
2: -8.5551338, -7.6554108, -8.5551338, -7.6554108, -0.5378416, 0.5368609
3: -3.3115957, -2.1908960, -3.3115957, -2.1908960, -0.6489437, 0.6499858
4: -10.5667381, -9.2098618, -10.5667381, -9.2098618, -1.1083078, 1.1102362
5: 8.1148443, 8.8970356, 8.1148443, 8.8970356, -0.6138463, 0.6113629
6: -7.1409636, -5.8589849, -7.1409636, -5.8589849, -0.6504970, 0.6532388
7: -12.1724968, -10.7043152, -12.1724968, -10.7043152, -1.0999012, 1.1020718
8: -1.9681163, -1.1633472, -1.9681163, -1.1633472, -0.5352530, 0.5357616
9: -3.4103088, -2.5724673, -3.4103088, -2.5724673, -0.7889593, 0.7889099

Time for backsubstitution: 23.08 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.83 + 543.51 = 600.34 seconds
