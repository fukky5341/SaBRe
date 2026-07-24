## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 147.6105270206


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-36.8120155, 131.9037170, -36.8120155, 131.9037170, -168.7157288, 168.7157288)
1: (-23.1996574, 80.0173416, -23.1996574, 80.0173416, -103.2169952, 103.2169952)
2: (-12.8174868, 74.0721664, -12.8174868, 74.0721664, -86.8896484, 86.8896484)
3: (-17.7853260, 109.6663971, -17.7853260, 109.6663971, -127.4516907, 127.4516907)
4: (-24.4155865, 90.5752716, -24.4155865, 90.5752716, -114.9908447, 114.9908447)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.90 + 1.88 = 3.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -147.9063397, upper bound: 147.9063397

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8979722, upper bound: 147.9031464
time: 0.69 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8979722, upper bound: 147.9063397
time: 0.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.45 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.45
Output dim: 0, lower bound: -147.8979722, upper bound: 147.9031464
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.45
Output dim: 0, lower bound: -147.8979722, upper bound: 147.9063397

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -32.5395088, 116.6438751, -35.8030434, 128.2714386, -160.8109436, 152.4469147
1: -20.3376465, 70.7146835, -22.5221405, 77.7991486, -98.1367950, 93.2368240
2: -11.1694107, 65.6986237, -12.4292040, 72.0705566, -83.2399673, 78.1278152
3: -15.5336437, 96.7999268, -17.2579327, 106.5969086, -122.1305542, 114.0578613
4: -21.2146854, 80.1311874, -23.6581402, 88.0792694, -109.2939529, 103.7893143

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8968961, upper bound: 147.8968961
time: 0.54 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8968961, upper bound: 147.9031464
time: 0.75 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -35.8220863, 128.3020172, -36.8120155, 131.9037170, -167.7257843, 165.1140289
1: -22.5247383, 77.7727814, -23.1996574, 80.0173416, -102.5420837, 100.9724426
2: -12.4312849, 72.0338287, -12.8174868, 74.0721664, -86.5034485, 84.8513107
3: -17.2587471, 106.5684586, -17.7853260, 109.6663971, -126.9251099, 124.3537827
4: -23.6584034, 88.0513687, -24.4155865, 90.5752716, -114.2336578, 112.4669495

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9031464, upper bound: 147.8979722
time: 0.67 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9031464, upper bound: 147.9063397
time: 0.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.11 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -147.8968961, upper bound: 147.8968961
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -147.8968961, upper bound: 147.9031464
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -147.9031464, upper bound: 147.8979722
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 0, lower bound: -147.9031464, upper bound: 147.9063397

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -32.5395088, 116.6438751, -32.5395088, 116.6438751, -149.1833801, 149.1833801
1: -20.3376465, 70.7146835, -20.3376465, 70.7146835, -91.0523300, 91.0523300
2: -11.1694107, 65.6986237, -11.1694107, 65.6986237, -76.8680344, 76.8680344
3: -15.5336437, 96.7999268, -15.5336437, 96.7999268, -112.3335724, 112.3335724
4: -21.2146854, 80.1311874, -21.2146854, 80.1311874, -101.3458710, 101.3458710

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8946302, upper bound: 147.8885052
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8878647, upper bound: 147.8878647
time: 0.51 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -32.5395088, 116.6438751, -35.8220863, 128.3020172, -160.8415222, 152.4659424
1: -20.3376465, 70.7146835, -22.5247383, 77.7727814, -98.1104279, 93.2394257
2: -11.1694107, 65.6986237, -12.4312849, 72.0338287, -83.2032394, 78.1298981
3: -15.5336437, 96.7999268, -17.2587471, 106.5684586, -122.1021042, 114.0586700
4: -21.2146854, 80.1311874, -23.6584034, 88.0513687, -109.2660522, 103.7895813

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8889976, upper bound: 147.8799256
time: 0.60 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8889976, upper bound: 147.9027298
time: 0.68 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -35.8220863, 128.3020172, -32.5395088, 116.6438751, -152.4659424, 160.8415222
1: -22.5247383, 77.7727814, -20.3376465, 70.7146835, -93.2394257, 98.1104279
2: -12.4312849, 72.0338287, -11.1694107, 65.6986237, -78.1298981, 83.2032394
3: -17.2587471, 106.5684586, -15.5336437, 96.7999268, -114.0586700, 122.1021042
4: -23.6584034, 88.0513687, -21.2146854, 80.1311874, -103.7895813, 109.2660522

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8799256, upper bound: 147.8889976
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9027298, upper bound: 147.8974282
time: 0.81 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -35.8220863, 128.3020172, -35.8220863, 128.3020172, -164.1240845, 164.1240845
1: -22.5247383, 77.7727814, -22.5247383, 77.7727814, -100.2975159, 100.2975159
2: -12.4312849, 72.0338287, -12.4312849, 72.0338287, -84.4651108, 84.4651108
3: -17.2587471, 106.5684586, -17.2587471, 106.5684586, -123.8271942, 123.8271942
4: -23.6584034, 88.0513687, -23.6584034, 88.0513687, -111.7097626, 111.7097626

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8889976, upper bound: 147.8965721
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.9027298, upper bound: 147.9059917
time: 0.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.21 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -147.8946302, upper bound: 147.8885052
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -147.8878647, upper bound: 147.8878647
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -147.8889976, upper bound: 147.8799256
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -147.8889976, upper bound: 147.9027298
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -147.8799256, upper bound: 147.8889976
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -147.9027298, upper bound: 147.8974282
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -147.8889976, upper bound: 147.8965721
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.21
Output dim: 0, lower bound: -147.9027298, upper bound: 147.9059917

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -31.4465733, 112.7952576, -32.5395088, 116.6438751, -148.0904388, 145.3347626
1: -19.6368332, 68.2494278, -20.3376465, 70.7146835, -90.3515167, 88.5870743
2: -10.7724228, 63.4087257, -11.1694107, 65.6986237, -76.4710312, 74.5781403
3: -14.9969807, 93.4350967, -15.5336437, 96.7999268, -111.7969055, 108.9687424
4: -20.4533501, 77.3404007, -21.2146854, 80.1311874, -100.5845337, 98.5550842

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8878647, upper bound: 147.8878647
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8878647, upper bound: 147.8878647
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -31.6359653, 114.2023163, -32.1392899, 115.2060318, -146.8419952, 146.3416138
1: -19.6855717, 68.7788773, -20.0670891, 69.8024445, -89.4880142, 88.8459549
2: -10.7376862, 64.0317841, -11.0151033, 64.8590775, -75.5967484, 75.0468903
3: -14.9771919, 94.0270538, -15.3243179, 95.5434036, -110.5205994, 109.3513718
4: -20.4308472, 77.9048691, -20.9212456, 79.0922699, -99.5231171, 98.8260956

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8864054, upper bound: 147.8857998
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8864823, upper bound: 147.8864780
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -29.9296894, 106.8018188, -28.7831535, 102.2729568, -132.2026215, 135.5849609
1: -18.6001244, 64.5892639, -18.0049553, 62.0602341, -80.6603394, 82.5942078
2: -10.2082996, 60.0025749, -9.9121733, 57.5451241, -67.7534256, 69.9147339
3: -14.2213764, 88.4619293, -13.7969894, 85.1259155, -99.3472824, 102.2589188
4: -19.3484688, 73.1995850, -18.8322620, 70.3497391, -89.6982117, 92.0318451

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8707870, upper bound: 147.8799256
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8735870, upper bound: 147.8799256
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -32.3503723, 115.9600372, -34.7859802, 124.5401306, -156.8905029, 150.7460175
1: -20.2141857, 70.3006134, -21.8452053, 75.5044785, -95.7186661, 92.1458206
2: -11.0998955, 65.3210831, -12.0506525, 69.9723740, -81.0722656, 77.3717270
3: -15.4386683, 96.2273483, -16.7325726, 103.4252777, -118.8639450, 112.9599228
4: -21.0817699, 79.6635361, -22.9276199, 85.4843063, -106.5660629, 102.5911560

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8742353, upper bound: 147.8912723
time: 0.90 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8756750, upper bound: 147.9027298
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -28.7831535, 102.2729568, -29.9296894, 106.8018188, -135.5849609, 132.2026215
1: -18.0049553, 62.0602341, -18.6001244, 64.5892639, -82.5942078, 80.6603394
2: -9.9121733, 57.5451241, -10.2082996, 60.0025749, -69.9147339, 67.7534256
3: -13.7969894, 85.1259155, -14.2213764, 88.4619293, -102.2589188, 99.3472824
4: -18.8322620, 70.3497391, -19.3484688, 73.1995850, -92.0318451, 89.6982117

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8799256, upper bound: 147.8735870
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8799256, upper bound: 147.8889976
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -34.7859802, 124.5401306, -32.3503723, 115.9600372, -150.7460175, 156.8905029
1: -21.8452053, 75.5044785, -20.2141857, 70.3006134, -92.1458206, 95.7186661
2: -12.0506525, 69.9723740, -11.0998955, 65.3210831, -77.3717270, 81.0722656
3: -16.7325726, 103.4252777, -15.4386683, 96.2273483, -112.9599228, 118.8639450
4: -22.9276199, 85.4843063, -21.0817699, 79.6635361, -102.5911560, 106.5660629

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8912723, upper bound: 147.8756750
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8912723, upper bound: 147.8974282
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -32.9254036, 117.2953033, -28.7831535, 102.2729568, -135.1983185, 146.0784302
1: -20.6097698, 71.2012863, -18.0049553, 62.0602341, -82.6699905, 89.2062378
2: -11.3723116, 65.9594803, -9.9121733, 57.5451241, -68.9174347, 75.8716354
3: -15.8014431, 97.5625610, -13.7969894, 85.1259155, -100.9273605, 111.3595505
4: -21.6213665, 80.6075516, -18.8322620, 70.3497391, -91.9711075, 99.4398117

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8707870, upper bound: 147.8952736
time: 0.56 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8707870, upper bound: 147.8965721
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -35.6146469, 127.5243073, -34.7859802, 124.5401306, -160.1547852, 162.3102875
1: -22.3874702, 77.3188705, -21.8452053, 75.5044785, -97.8919525, 99.1640778
2: -12.3552151, 71.6226654, -12.0506525, 69.9723740, -82.3275909, 83.6733170
3: -17.1529312, 105.9364929, -16.7325726, 103.4252777, -120.5782089, 122.6690674
4: -23.5114002, 87.5381241, -22.9276199, 85.4843063, -108.9956894, 110.4657440

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8968029, upper bound: 147.9021338
time: 0.55 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8968029, upper bound: 147.9059917
time: 0.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.35 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -147.8878647, upper bound: 147.8878647
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -147.8878647, upper bound: 147.8878647
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -147.8864054, upper bound: 147.8857998
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -147.8864823, upper bound: 147.8864780
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -147.8707870, upper bound: 147.8799256
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -147.8735870, upper bound: 147.8799256
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -147.8742353, upper bound: 147.8912723
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -147.8756750, upper bound: 147.9027298
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -147.8799256, upper bound: 147.8735870
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -147.8799256, upper bound: 147.8889976
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -147.8912723, upper bound: 147.8756750
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -147.8912723, upper bound: 147.8974282
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -147.8707870, upper bound: 147.8952736
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -147.8707870, upper bound: 147.8965721
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -147.8968029, upper bound: 147.9021338
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 0, lower bound: -147.8968029, upper bound: 147.9059917

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -31.4465733, 112.7952576, -31.4465733, 112.7952576, -144.2418365, 144.2418365
1: -19.6368332, 68.2494278, -19.6368332, 68.2494278, -87.8862610, 87.8862610
2: -10.7724228, 63.4087257, -10.7724228, 63.4087257, -74.1811523, 74.1811523
3: -14.9969807, 93.4350967, -14.9969807, 93.4350967, -108.4320679, 108.4320755
4: -20.4533501, 77.3404007, -20.4533501, 77.3404007, -97.7937393, 97.7937393

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8834857, upper bound: 147.8839965
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8820694, upper bound: 147.8820694
time: 0.53 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -31.4465733, 112.7952576, -31.6359653, 114.2023163, -145.6488953, 144.4312286
1: -19.6368332, 68.2494278, -19.6855717, 68.7788773, -88.4157028, 87.9349976
2: -10.7724228, 63.4087257, -10.7376862, 64.0317841, -74.8042068, 74.1464081
3: -14.9969807, 93.4350967, -14.9771919, 94.0270538, -109.0240326, 108.4122772
4: -20.4533501, 77.3404007, -20.4308472, 77.9048691, -98.3582077, 97.7712402

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8923584, upper bound: 147.8846716
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8820694, upper bound: 147.8823061
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -31.3145618, 113.0948944, -29.5474529, 105.9353714, -137.2499390, 142.6423492
1: -19.4738026, 68.0635986, -18.3042164, 63.8095169, -83.2833176, 86.3678131
2: -10.6193581, 63.3704147, -10.0263863, 59.3140831, -69.9334412, 73.3968048
3: -14.8248777, 93.0379486, -14.0281839, 87.2608643, -102.0857391, 107.0661316
4: -20.2044754, 77.0934525, -19.0362892, 72.2960281, -92.5005035, 96.1297379

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8684905, upper bound: 147.8791833
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8843113, upper bound: 147.8836780
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -31.1166210, 112.4089355, -28.1810112, 101.3776855, -132.4943085, 140.5899048
1: -19.3483295, 67.6842880, -17.4782982, 60.7856102, -80.1339264, 85.1625824
2: -10.5509176, 63.0361786, -9.5763731, 56.4560699, -67.0069885, 72.6125488
3: -14.7306280, 92.5039139, -13.4392157, 83.2772827, -98.0079041, 105.9431305
4: -20.0720119, 76.6603699, -18.1366348, 68.8675385, -88.9395370, 94.7969971

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8858020, upper bound: 147.8864012
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8858020, upper bound: 147.8864823
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -26.3230553, 93.8408813, -28.7831535, 102.2729568, -128.5959625, 122.6240387
1: -16.3732510, 56.6407509, -18.0049553, 62.0602341, -78.4334793, 74.6457062
2: -8.9605923, 52.6873741, -9.9121733, 57.5451241, -66.5057144, 62.5995369
3: -12.5455513, 77.6787491, -13.7969894, 85.1259155, -97.6714630, 91.4757385
4: -16.9255257, 64.3158340, -18.8322620, 70.3497391, -87.2752533, 83.1480865

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8707462, upper bound: 147.8758691
time: 0.87 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8724155, upper bound: 147.8778044
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -31.4150181, 112.5959473, -28.7831535, 102.2729568, -133.6879578, 141.3790894
1: -19.6078739, 68.1838226, -18.0049553, 62.0602341, -81.6680984, 86.1887817
2: -10.7586298, 63.3652916, -9.9121733, 57.5451241, -68.3037567, 73.2774658
3: -14.9687634, 93.3341751, -13.7969894, 85.1259155, -100.0946808, 107.1311646
4: -20.4348564, 77.2663956, -18.8322620, 70.3497391, -90.7845764, 96.0986557

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8726334, upper bound: 147.8792249
time: 0.58 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8724155, upper bound: 147.8778044
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -26.3230553, 93.8408813, -34.7859802, 124.5401306, -150.8631744, 128.6268616
1: -16.3732510, 56.6407509, -21.8452053, 75.5044785, -91.8777313, 78.4859543
2: -8.9605923, 52.6873741, -12.0506525, 69.9723740, -78.9329681, 64.7380295
3: -12.5455513, 77.6787491, -16.7325726, 103.4252777, -115.9708252, 94.4113235
4: -16.9255257, 64.3158340, -22.9276199, 85.4843063, -102.4098206, 87.2434540

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8707462, upper bound: 147.8908327
time: 0.76 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8724155, upper bound: 147.8908335
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -31.5998249, 113.2599030, -34.7859802, 124.5401306, -156.1399536, 148.0458832
1: -19.7260361, 68.6501465, -21.8452053, 75.5044785, -95.2305145, 90.4953537
2: -10.8250427, 63.8127785, -12.0506525, 69.9723740, -80.7974167, 75.8634338
3: -15.0623989, 93.9510422, -16.7325726, 103.4252777, -118.4876709, 110.6836166
4: -20.5557213, 77.7999344, -22.9276199, 85.4843063, -106.0400238, 100.7275543

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8707462, upper bound: 147.8917399
time: 0.54 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8724155, upper bound: 147.8995598
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -28.7831535, 102.2729568, -26.3230553, 93.8408813, -122.6240387, 128.5959625
1: -18.0049553, 62.0602341, -16.3732510, 56.6407509, -74.6457062, 78.4334793
2: -9.9121733, 57.5451241, -8.9605923, 52.6873741, -62.5995407, 66.5057144
3: -13.7969894, 85.1259155, -12.5455513, 77.6787491, -91.4757385, 97.6714630
4: -18.8322620, 70.3497391, -16.9255257, 64.3158340, -83.1480865, 87.2752533

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8758691, upper bound: 147.8707462
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8778044, upper bound: 147.8724155
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -28.7831535, 102.2729568, -31.4150181, 112.5959473, -141.3790894, 133.6879425
1: -18.0049553, 62.0602341, -19.6078739, 68.1838226, -86.1887817, 81.6680984
2: -9.9121733, 57.5451241, -10.7586298, 63.3652916, -73.2774658, 68.3037567
3: -13.7969894, 85.1259155, -14.9687634, 93.3341751, -107.1311646, 100.0946808
4: -18.8322620, 70.3497391, -20.4348564, 77.2663956, -96.0986557, 90.7845764

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8792249, upper bound: 147.8886539
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8778044, upper bound: 147.8879790
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -34.7859802, 124.5401306, -26.3230553, 93.8408813, -128.6268616, 150.8631744
1: -21.8452053, 75.5044785, -16.3732510, 56.6407509, -78.4859543, 91.8777313
2: -12.0506525, 69.9723740, -8.9605923, 52.6873741, -64.7380295, 78.9329681
3: -16.7325726, 103.4252777, -12.5455513, 77.6787491, -94.4113235, 115.9708252
4: -22.9276199, 85.4843063, -16.9255257, 64.3158340, -87.2434540, 102.4098206

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8872312, upper bound: 147.8732083
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8909456, upper bound: 147.8754726
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -34.7859802, 124.5401306, -31.5998249, 113.2599030, -148.0458679, 156.1399536
1: -21.8452053, 75.5044785, -19.7260361, 68.6501465, -90.4953537, 95.2305145
2: -12.0506525, 69.9723740, -10.8250427, 63.8127785, -75.8634262, 80.7974167
3: -16.7325726, 103.4252777, -15.0623989, 93.9510422, -110.6836166, 118.4876709
4: -22.9276199, 85.4843063, -20.5557213, 77.7999344, -100.7275543, 106.0400238

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8894808, upper bound: 147.8882394
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8822134, upper bound: 147.8855093
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -28.7831535, 102.2729568, -28.7831535, 102.2729568, -131.0560913, 131.0560913
1: -18.0049553, 62.0602341, -18.0049553, 62.0602341, -80.0651779, 80.0651779
2: -9.9121733, 57.5451241, -9.9121733, 57.5451241, -67.4572983, 67.4572983
3: -13.7969894, 85.1259155, -13.7969894, 85.1259155, -98.9229050, 98.9229050
4: -18.8322620, 70.3497391, -18.8322620, 70.3497391, -89.1819992, 89.1819992

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8916049, upper bound: 147.8872327
time: 0.73 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8861587, upper bound: 147.8861587
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -34.7859306, 124.5399475, -28.7831535, 102.2729568, -137.0588837, 153.3231049
1: -21.8451576, 75.5043640, -18.0049553, 62.0602341, -83.9053879, 93.5093155
2: -12.0506220, 69.9722824, -9.9121733, 57.5451241, -69.5957489, 79.8844528
3: -16.7325287, 103.4251099, -13.7969894, 85.1259155, -101.8584366, 117.2220993
4: -22.9275742, 85.4841614, -18.8322620, 70.3497391, -93.2773132, 104.3164215

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8882181, upper bound: 147.8857069
time: 0.55 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8825620, upper bound: 147.8860699
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -28.7831535, 102.2729568, -34.7859802, 124.5401306, -153.3232880, 137.0589294
1: -18.0049553, 62.0602341, -21.8452053, 75.5044785, -93.5094299, 83.9054413
2: -9.9121733, 57.5451241, -12.0506525, 69.9723740, -79.8845444, 69.5957794
3: -13.7969894, 85.1259155, -16.7325726, 103.4252777, -117.2222672, 101.8584900
4: -18.8322620, 70.3497391, -22.9276199, 85.4843063, -104.3165665, 93.2773590

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8825965, upper bound: 147.8849168
time: 0.54 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8825620, upper bound: 147.8941986
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -34.7859802, 124.5401306, -34.7859802, 124.5401306, -159.3261108, 159.3261108
1: -21.8452053, 75.5044785, -21.8452053, 75.5044785, -97.3496857, 97.3496857
2: -12.0506525, 69.9723740, -12.0506525, 69.9723740, -82.0230255, 82.0230255
3: -16.7325726, 103.4252777, -16.7325726, 103.4252777, -120.1578522, 120.1578522
4: -22.9276199, 85.4843063, -22.9276199, 85.4843063, -108.4119263, 108.4119263

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8826135, upper bound: 147.8886462
time: 0.88 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8822087, upper bound: 147.8869221
time: 0.79 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.69 seconds
NS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8834857, upper bound: 147.8839965
NS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8820694, upper bound: 147.8820694
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8923584, upper bound: 147.8846716
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8820694, upper bound: 147.8823061
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8684905, upper bound: 147.8791833
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8843113, upper bound: 147.8836780
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8858020, upper bound: 147.8864012
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8858020, upper bound: 147.8864823
NS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8707462, upper bound: 147.8758691
NS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8724155, upper bound: 147.8778044
NS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8726334, upper bound: 147.8792249
NS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8724155, upper bound: 147.8778044
NS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8707462, upper bound: 147.8908327
NS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8724155, upper bound: 147.8908335
NS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8707462, upper bound: 147.8917399
NS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8724155, upper bound: 147.8995598
NS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8758691, upper bound: 147.8707462
NS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8778044, upper bound: 147.8724155
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8792249, upper bound: 147.8886539
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8778044, upper bound: 147.8879790
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8872312, upper bound: 147.8732083
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8909456, upper bound: 147.8754726
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8894808, upper bound: 147.8882394
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8822134, upper bound: 147.8855093
NS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8916049, upper bound: 147.8872327
NS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8861587, upper bound: 147.8861587
NS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8882181, upper bound: 147.8857069
NS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8825620, upper bound: 147.8860699
NS_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8825965, upper bound: 147.8849168
NS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8825620, upper bound: 147.8941986
NS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8826135, upper bound: 147.8886462
NS_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -147.8822087, upper bound: 147.8869221

## BFS NS instance: NS_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -31.4465733, 112.7952576, -30.8583736, 110.6469421, -142.0935059, 143.6536255
1: -19.6368332, 68.2494278, -19.2477093, 66.8586197, -86.4954453, 87.4971390
2: -10.7724228, 63.4087257, -10.5548000, 62.1071167, -72.8795319, 73.9635239
3: -14.9969807, 93.4350967, -14.7066040, 91.5520782, -106.5490417, 108.1417007
4: -20.4533501, 77.3404007, -20.0339489, 75.7693481, -96.2226868, 97.3743439

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8820694, upper bound: 147.8820694
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8820694, upper bound: 147.8820694
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -30.8859482, 110.7339630, -29.1225548, 104.1819839, -135.0679321, 139.8565216
1: -19.2563820, 66.9321899, -18.0245533, 62.6214600, -81.8778381, 84.9567413
2: -10.5579195, 62.1893578, -9.8561020, 58.1786079, -68.7365265, 72.0454559
3: -14.7113228, 91.6351929, -13.7885504, 85.7612534, -100.4725800, 105.4237442
4: -20.0377407, 75.8497467, -18.6663570, 70.9539566, -90.9916992, 94.5161057

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8820694, upper bound: 147.8820694
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8820694, upper bound: 147.8820694
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -30.8583736, 110.6469421, -31.6359653, 114.2023163, -145.0606842, 142.2828979
1: -19.2477093, 66.8586197, -19.6855717, 68.7788773, -88.0265884, 86.5441818
2: -10.5548000, 62.1071167, -10.7376862, 64.0317841, -74.5865860, 72.8448029
3: -14.7066040, 91.5520782, -14.9771919, 94.0270538, -108.7336578, 106.5292587
4: -20.0339489, 75.7693481, -20.4308472, 77.9048691, -97.9388046, 96.2001877

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8828197, upper bound: 147.8823061
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8828197, upper bound: 147.8823061
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -29.1225548, 104.1819839, -31.2126122, 112.6362534, -141.7588043, 135.3945923
1: -18.0245533, 62.6214600, -19.3981380, 67.8140793, -85.8386307, 82.0195923
2: -9.8561020, 58.1786079, -10.5758305, 63.1467476, -73.0028458, 68.7544403
3: -13.7885504, 85.7612534, -14.7665644, 92.6994171, -106.4879684, 100.5278168
4: -18.6663570, 70.9539566, -20.1142673, 76.8134689, -95.4798279, 91.0682220

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8828197, upper bound: 147.8823061
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8828197, upper bound: 147.8823061
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -24.2356567, 86.2994080, -27.1588058, 96.6940308, -120.9296875, 113.4581985
1: -14.9194565, 51.9469337, -16.6791821, 58.1066780, -73.0261307, 68.6261139
2: -8.1108665, 48.4577065, -9.1280499, 54.0097313, -62.1205978, 57.5857506
3: -11.4884386, 71.1097183, -12.8320627, 79.4705353, -90.9589691, 83.9417801
4: -15.3013477, 59.0040169, -17.2855644, 65.8328934, -81.1342392, 76.2895813

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8683703, upper bound: 147.8791833
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8683703, upper bound: 147.8791833
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -30.4215851, 109.9125900, -29.3509502, 105.1730804, -135.5946655, 139.2635193
1: -18.8916531, 66.1009903, -18.1730442, 63.3770485, -82.2686996, 84.2740326
2: -10.2924824, 61.5798836, -9.9543543, 58.9205208, -69.2129898, 71.5342331
3: -14.3762436, 90.3226852, -13.9306192, 86.6603317, -101.0365753, 104.2533035
4: -19.5791988, 74.8753128, -18.8980999, 71.8085327, -91.3877258, 93.7734070

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8799638, upper bound: 147.8690277
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8799638, upper bound: 147.8836780
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -29.4338264, 106.4715195, -28.1810112, 101.3776855, -130.8114777, 134.6525116
1: -18.2131405, 63.7938576, -17.4782982, 60.7856102, -78.9987411, 81.2721558
2: -9.9132318, 59.4197426, -9.5763731, 56.4560699, -66.3693008, 68.9961014
3: -13.9110613, 87.1336899, -13.4392157, 83.2772827, -97.1883316, 100.5729065
4: -18.8489304, 72.2434464, -18.1366348, 68.8675385, -87.7164612, 90.3800659

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8683703, upper bound: 147.8799624
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8835426, upper bound: 147.8843113
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -27.6878624, 99.8279800, -28.1810112, 101.3776855, -129.0655518, 128.0089874
1: -17.1215916, 59.7900085, -17.4782982, 60.7856102, -77.9071960, 77.2683105
2: -9.3374052, 55.6133690, -9.5763731, 56.4560699, -65.7934723, 65.1897278
3: -13.1690903, 81.8129196, -13.4392157, 83.2772827, -96.4463654, 95.2521362
4: -17.7071171, 67.7328186, -18.1366348, 68.8675385, -86.5746536, 85.8694305

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8683703, upper bound: 147.8797182
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8835426, upper bound: 147.8844207
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -23.4658470, 83.4173279, -28.3024120, 100.5276260, -123.9934692, 111.7197418
1: -14.4133177, 49.9048729, -17.6813965, 60.9720078, -75.3853226, 67.5862732
2: -7.8627238, 46.4424973, -9.7306776, 56.5443153, -64.4070206, 56.1731758
3: -11.1510811, 68.3978577, -13.5614939, 83.6218643, -94.7729492, 81.9593506
4: -14.8063288, 56.7009087, -18.4833508, 69.1213303, -83.9276505, 75.1842575

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B1_A1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8707462, upper bound: 147.8778965
time: 0.57 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8707462, upper bound: 147.8778965
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -24.5587463, 88.0569992, -28.2285061, 100.2849426, -124.8436890, 116.2855072
1: -15.1891947, 52.6221390, -17.6403503, 60.8724289, -76.0616226, 70.2624893
2: -8.2939606, 48.9453201, -9.7110348, 56.4602509, -64.7542114, 58.6563568
3: -11.7343845, 72.1928177, -13.5314474, 83.4800186, -95.2144012, 85.7242661
4: -15.6375437, 59.7779083, -18.4459629, 69.0058289, -84.6433640, 78.2238693

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B1_A1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8698256, upper bound: 147.8781238
time: 0.74 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8698256, upper bound: 147.8781238
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -31.0206642, 111.1684189, -25.4176807, 90.1015778, -121.1222382, 136.5861053
1: -19.3398418, 67.2794266, -15.7430477, 54.4464111, -73.7862473, 83.0224686
2: -10.6092854, 62.5311661, -8.6457529, 50.5347137, -61.1439934, 71.1769180
3: -14.7731094, 92.0832748, -12.1659594, 74.6058578, -89.3789673, 104.2492218
4: -20.1488075, 76.2441406, -16.3946724, 61.7497253, -81.8985291, 92.6388016

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8794580, upper bound: 147.8758691
time: 0.76 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8794580, upper bound: 147.8778044
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -30.8560867, 110.6500168, -26.7695236, 95.5459671, -126.4020538, 137.4195404
1: -19.2481003, 66.9861526, -16.7035770, 57.6046410, -76.8527374, 83.6897278
2: -10.5607405, 62.2629547, -9.1787004, 53.3813210, -63.9420586, 71.4416351
3: -14.7071486, 91.6838608, -12.8916302, 79.0410767, -93.7482224, 104.5754776
4: -20.0559349, 75.9089966, -17.4574413, 65.3044891, -85.3604050, 93.3664246

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8794580, upper bound: 147.8758691
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8794580, upper bound: 147.8778044
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -23.4658470, 83.4173279, -34.3430595, 122.9739761, -146.4398193, 117.7603912
1: -14.4133177, 49.9048729, -21.5546455, 74.5252151, -88.9385223, 71.4595184
2: -7.8627238, 46.4424973, -11.8872938, 69.0717087, -76.9344101, 58.3297920
3: -11.1510811, 68.3978577, -16.5158272, 102.0692978, -113.2203827, 84.9136810
4: -14.8063288, 56.7009087, -22.6147785, 84.3762054, -99.1825333, 79.3156891

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8692636, upper bound: 147.8841929
time: 0.78 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8734367, upper bound: 147.8897022
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -24.5587463, 88.0569992, -34.1918449, 122.4774170, -147.0361633, 122.2488403
1: -15.1891947, 52.6221390, -21.4631538, 74.2344284, -89.4236145, 74.0852966
2: -8.2939606, 48.9453201, -11.8383951, 68.8089523, -77.1029129, 60.7837143
3: -11.7343845, 72.1928177, -16.4494553, 101.6695938, -113.4039764, 88.6422729
4: -15.6375437, 59.7779083, -22.5226746, 84.0463257, -99.6838684, 82.3005829

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8735546, upper bound: 147.8875575
time: 0.78 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8743754, upper bound: 147.8897435
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -28.9132442, 103.5669708, -34.3430595, 122.9739761, -151.8872070, 137.9100342
1: -17.8996487, 62.4445076, -21.5546455, 74.5252151, -92.4248581, 83.9991379
2: -9.8057909, 58.0699806, -11.8872938, 69.0717087, -78.8774872, 69.9572754
3: -13.7253065, 85.3750687, -16.5158272, 102.0692978, -115.7946014, 101.8908997
4: -18.6112766, 70.7665939, -22.6147785, 84.3762054, -102.9874649, 93.3813705

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8880877, upper bound: 147.8917399
time: 0.69 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8880877, upper bound: 147.8917399
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -26.9912338, 96.3910828, -34.1918449, 122.4774170, -149.4686584, 130.5829315
1: -16.7048092, 58.2303963, -21.4631538, 74.2344284, -90.9392395, 79.6935501
2: -9.1734991, 54.1287651, -11.8383951, 68.8089523, -77.9824524, 65.9671478
3: -12.8568783, 79.7242203, -16.4494553, 101.6695938, -114.5264740, 96.1736755
4: -17.3538589, 65.9859009, -22.5226746, 84.0463257, -101.4001846, 88.5085678

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8851207, upper bound: 147.8906318
time: 0.71 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8805325, upper bound: 147.8801038
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -28.3024120, 100.5276260, -23.4658470, 83.4173279, -111.7197418, 123.9934692
1: -17.6813965, 60.9720078, -14.4133177, 49.9048729, -67.5862732, 75.3853149
2: -9.7306776, 56.5443153, -7.8627238, 46.4424973, -56.1731758, 64.4070206
3: -13.5614939, 83.6218643, -11.1510811, 68.3978577, -81.9593506, 94.7729492
4: -18.4833508, 69.1213303, -14.8063288, 56.7009087, -75.1842575, 83.9276505

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8778965, upper bound: 147.8707462
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8778965, upper bound: 147.8707462
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -28.2285061, 100.2849426, -24.5587463, 88.0569992, -116.2855072, 124.8436890
1: -17.6403503, 60.8724289, -15.1891947, 52.6221390, -70.2624893, 76.0616226
2: -9.7110348, 56.4602509, -8.2939606, 48.9453201, -58.6563568, 64.7542114
3: -13.5314474, 83.4800186, -11.7343845, 72.1928177, -85.7242661, 95.2144012
4: -18.4459629, 69.0058289, -15.6375437, 59.7779083, -78.2238693, 84.6433716

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8781238, upper bound: 147.8724155
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8781238, upper bound: 147.8724155
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -25.4176807, 90.1015778, -31.0206642, 111.1684189, -136.5861053, 121.1222382
1: -15.7430477, 54.4464111, -19.3398418, 67.2794266, -83.0224686, 73.7862473
2: -8.6457529, 50.5347137, -10.6092854, 62.5311661, -71.1769180, 61.1439972
3: -12.1659594, 74.6058578, -14.7731094, 92.0832748, -104.2492218, 89.3789673
4: -16.3946724, 61.7497253, -20.1488075, 76.2441406, -92.6388016, 81.8985291

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8758691, upper bound: 147.8794580
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8758691, upper bound: 147.8879790
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -26.7695236, 95.5459671, -30.8560867, 110.6500168, -137.4195404, 126.4020538
1: -16.7035770, 57.6046410, -19.2481003, 66.9861526, -83.6897278, 76.8527374
2: -9.1787004, 53.3813210, -10.5607405, 62.2629547, -71.4416428, 63.9420509
3: -12.8916302, 79.0410767, -14.7071486, 91.6838608, -104.5754776, 93.7482224
4: -17.4574413, 65.3044891, -20.0559349, 75.9089966, -93.3664246, 85.3604126

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8758691, upper bound: 147.8794580
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8758691, upper bound: 147.8879790
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -21.9257565, 77.5900879, -22.5560932, 80.2381134, -102.1638565, 100.1461792
1: -13.5214148, 45.7028389, -13.9220753, 47.5371437, -61.0585594, 59.6249161
2: -7.5008678, 41.6937523, -7.5722055, 44.0365028, -51.5373688, 49.2659569
3: -10.4911938, 63.2662544, -10.7038975, 65.4212952, -75.9124908, 73.9701309
4: -14.2142792, 51.4617996, -14.2416945, 53.9410515, -68.1553345, 65.7034912

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8863509, upper bound: 147.8715598
time: 0.51 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8832172, upper bound: 147.8686071
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 22

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -31.5629597, 112.4799652, -25.5646248, 91.0948792, -122.6578369, 138.0445709
1: -19.6963806, 68.3293991, -15.8676863, 54.9327087, -74.6290894, 84.1970673
2: -10.8276711, 63.3917847, -8.6620951, 51.1278191, -61.9554901, 72.0538712
3: -15.0999479, 93.5080948, -12.1662264, 75.3144836, -90.4144287, 105.6743240
4: -20.5950108, 77.3433151, -16.3564758, 62.3885574, -82.9835663, 93.6997757

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8880136, upper bound: 147.8734294
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8898910, upper bound: 147.8752249
time: 0.52 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -33.6123505, 120.4252777, -31.5998249, 113.2599030, -146.8722534, 152.0250702
1: -21.0901127, 72.8470917, -19.7260361, 68.6501465, -89.7402573, 92.5731201
2: -11.6257772, 67.4980087, -10.8250427, 63.8127785, -75.4385376, 78.3230515
3: -16.1589336, 99.7994843, -15.0623989, 93.9510422, -110.1099625, 114.8618851
4: -22.1145821, 82.4763489, -20.5557213, 77.7999344, -99.9145050, 103.0320740

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8865328, upper bound: 147.8855092
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8865328, upper bound: 147.8855093
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -33.6332817, 121.4531097, -31.2120705, 111.8100586, -145.4433289, 152.6651611
1: -21.0786572, 73.0833740, -19.4574795, 67.7586060, -88.8372574, 92.5408554
2: -11.5485439, 67.8458862, -10.6726656, 62.9977989, -74.5463409, 78.5185547
3: -16.0521793, 100.0187302, -14.8566809, 92.7179184, -108.7700882, 114.8754120
4: -22.0051498, 82.7202911, -20.2658272, 76.7862167, -98.7913589, 102.9861145

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8786183, upper bound: 147.8688736
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8864852, upper bound: 147.8854277
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -18.9916611, 66.8387070, -24.8128281, 88.1222534, -107.1139145, 91.6515350
1: -11.6649609, 38.9399948, -15.4690657, 52.5646133, -64.2295761, 54.4090500
2: -6.4185624, 35.4893227, -8.4777250, 48.4877243, -54.9062843, 43.9670334
3: -9.0818567, 54.1079369, -11.9163389, 72.3882523, -81.4701004, 66.0242767
4: -12.0993586, 43.9316406, -16.0979462, 59.5549278, -71.6542816, 60.0295868

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8861587, upper bound: 147.8861587
time: 0.56 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8861587, upper bound: 147.8861587
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -26.2541981, 93.0125351, -27.9891968, 99.3892975, -125.6434937, 121.0017319
1: -16.2923088, 56.3743286, -17.4655457, 60.2975311, -76.5898361, 73.8398743
2: -8.9113579, 52.3589554, -9.5953732, 55.9468689, -64.8582153, 61.9543304
3: -12.5040159, 77.2152634, -13.3920021, 82.6631775, -95.1671906, 90.6072693
4: -16.9262562, 63.8913345, -18.2331066, 68.3472137, -85.2734680, 82.1244278

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8861587, upper bound: 147.8861587
time: 0.70 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8861587, upper bound: 147.8861587
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -30.1565361, 106.9011688, -27.8506317, 98.7290649, -128.8856049, 134.7518005
1: -18.7368374, 65.1004562, -17.3759880, 59.9809875, -78.7178268, 82.4764404
2: -10.3332443, 60.4726181, -9.5604010, 55.6559677, -65.9892120, 70.0330048
3: -14.4177284, 89.0045471, -13.3384380, 82.2311401, -96.6488647, 102.3429642
4: -19.6149368, 73.7423630, -18.1520710, 68.0054932, -87.6204147, 91.8944321

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8963931, upper bound: 147.8856085
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8963931, upper bound: 147.8856085
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -50.8791389, 188.7596588, -27.7888660, 98.6470108, -149.5261383, 216.5485229
1: -32.9595070, 110.1463470, -17.3122368, 59.7928505, -92.7523575, 127.4585724
2: -18.0147324, 100.8761215, -9.5205669, 55.4703293, -73.4850540, 110.3966827
3: -24.9221153, 152.7215881, -13.2940264, 81.9801865, -106.9022980, 166.0156097
4: -34.2294846, 124.6979828, -18.0737267, 67.7746124, -102.0040970, 142.7717133

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8966221, upper bound: 147.8860699
time: 0.67 seconds

## Relational analysis of NS_A2_B2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8966221, upper bound: 147.8860699
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -27.8506317, 98.7290649, -30.1565361, 106.9011688, -134.7518005, 128.8856049
1: -17.3759880, 59.9809875, -18.7368374, 65.1004562, -82.4764404, 78.7178268
2: -9.5604010, 55.6559677, -10.3332443, 60.4726181, -70.0330048, 65.9892044
3: -13.3384380, 82.2311401, -14.4177284, 89.0045471, -102.3429642, 96.6488647
4: -18.1520710, 68.0054932, -19.6149368, 73.7423630, -91.8944321, 87.6204147

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8831240, upper bound: 147.8849168
time: 0.61 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8831240, upper bound: 147.8849168
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -27.7888660, 98.6470108, -50.8792877, 188.7601624, -216.5490265, 149.5262909
1: -17.3122368, 59.7928505, -32.9596291, 110.1466522, -127.4588928, 92.7524796
2: -9.5205669, 55.4703293, -18.0148106, 100.8763885, -110.3969574, 73.4851379
3: -13.2940264, 81.9801865, -24.9222260, 152.7220306, -166.0160522, 106.9024124
4: -18.0737267, 67.7746124, -34.2296181, 124.6983719, -142.7720947, 102.0042267

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8851907, upper bound: 147.8941986
time: 0.79 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8851907, upper bound: 147.8941986
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -34.7859802, 124.5401306, -33.6123505, 120.4252777, -155.2112579, 158.1524811
1: -21.8452053, 75.5044785, -21.0901127, 72.8470917, -94.6922913, 96.5945740
2: -12.0506525, 69.9723740, -11.6257772, 67.4980087, -79.5486603, 81.5981522
3: -16.7325726, 103.4252777, -16.1589336, 99.7994843, -116.5320587, 119.5841980
4: -22.9276199, 85.4843063, -22.1145821, 82.4763489, -105.4039688, 107.5988846

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8840623, upper bound: 147.8854691
time: 0.61 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8867955, upper bound: 147.8879984
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -34.3877983, 123.0876694, -33.6332817, 121.4531097, -155.8409119, 156.7209320
1: -21.5764294, 74.6196060, -21.0786572, 73.0833740, -94.6597977, 95.6982651
2: -11.8969660, 69.1659088, -11.5485439, 67.8458862, -79.7428436, 80.7144547
3: -16.5224495, 102.1957397, -16.0521793, 100.0187302, -116.5411682, 118.2479172
4: -22.6341381, 84.4771805, -22.0051498, 82.7202911, -105.3544312, 106.4823303

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8839988, upper bound: 147.8839988
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8863933, upper bound: 147.8863293
time: 0.96 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.80 seconds
NS_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8820694, upper bound: 147.8820694
NS_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8820694, upper bound: 147.8820694
NS_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8820694, upper bound: 147.8820694
NS_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8820694, upper bound: 147.8820694
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8828197, upper bound: 147.8823061
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8828197, upper bound: 147.8823061
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8828197, upper bound: 147.8823061
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8828197, upper bound: 147.8823061
NS_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8683703, upper bound: 147.8791833
NS_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8683703, upper bound: 147.8791833
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8799638, upper bound: 147.8690277
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8799638, upper bound: 147.8836780
NS_A1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8683703, upper bound: 147.8799624
NS_A1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8835426, upper bound: 147.8843113
NS_A1_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8683703, upper bound: 147.8797182
NS_A1_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8835426, upper bound: 147.8844207
NS_A1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8707462, upper bound: 147.8778965
NS_A1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8707462, upper bound: 147.8778965
NS_A1_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8698256, upper bound: 147.8781238
NS_A1_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8698256, upper bound: 147.8781238
NS_A1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8794580, upper bound: 147.8758691
NS_A1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8794580, upper bound: 147.8778044
NS_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8794580, upper bound: 147.8758691
NS_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8794580, upper bound: 147.8778044
NS_A1_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8692636, upper bound: 147.8841929
NS_A1_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8734367, upper bound: 147.8897022
NS_A1_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8735546, upper bound: 147.8875575
NS_A1_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8743754, upper bound: 147.8897435
NS_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8880877, upper bound: 147.8917399
NS_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8880877, upper bound: 147.8917399
NS_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8851207, upper bound: 147.8906318
NS_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8805325, upper bound: 147.8801038
NS_A2_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8778965, upper bound: 147.8707462
NS_A2_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8778965, upper bound: 147.8707462
NS_A2_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8781238, upper bound: 147.8724155
NS_A2_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8781238, upper bound: 147.8724155
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8758691, upper bound: 147.8794580
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8758691, upper bound: 147.8879790
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8758691, upper bound: 147.8794580
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8758691, upper bound: 147.8879790
NS_A2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8880136, upper bound: 147.8734294
NS_A2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8898910, upper bound: 147.8752249
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8865328, upper bound: 147.8855092
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8865328, upper bound: 147.8855093
NS_A2_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8786183, upper bound: 147.8688736
NS_A2_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8864852, upper bound: 147.8854277
NS_A2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8861587, upper bound: 147.8861587
NS_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8861587, upper bound: 147.8861587
NS_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8861587, upper bound: 147.8861587
NS_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8861587, upper bound: 147.8861587
NS_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8963931, upper bound: 147.8856085
NS_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8963931, upper bound: 147.8856085
NS_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8966221, upper bound: 147.8860699
NS_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8966221, upper bound: 147.8860699
NS_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8831240, upper bound: 147.8849168
NS_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8831240, upper bound: 147.8849168
NS_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8851907, upper bound: 147.8941986
NS_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8851907, upper bound: 147.8941986
NS_A2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8840623, upper bound: 147.8854691
NS_A2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8867955, upper bound: 147.8879984
NS_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8839988, upper bound: 147.8839988
NS_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 0, lower bound: -147.8863933, upper bound: 147.8863293

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -30.8583736, 110.6469421, -30.8583736, 110.6469421, -141.5053101, 141.5053101
1: -19.2477093, 66.8586197, -19.2477093, 66.8586197, -86.1063232, 86.1063232
2: -10.5548000, 62.1071167, -10.5548000, 62.1071167, -72.6619186, 72.6619186
3: -14.7066040, 91.5520782, -14.7066040, 91.5520782, -106.2586823, 106.2586823
4: -20.0339489, 75.7693481, -20.0339489, 75.7693481, -95.8032913, 95.8032913

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -29.1225548, 104.1819839, -30.8583736, 110.6469421, -139.7695007, 135.0403595
1: -18.0245533, 62.6214600, -19.2477093, 66.8586197, -84.8831635, 81.8691711
2: -9.8561020, 58.1786079, -10.5548000, 62.1071167, -71.9632187, 68.7334061
3: -13.7885504, 85.7612534, -14.7066040, 91.5520782, -105.3406219, 100.4678574
4: -18.6663570, 70.9539566, -20.0339489, 75.7693481, -94.4357071, 90.9878998

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -30.8583736, 110.6469421, -29.1225548, 104.1819839, -135.0403595, 139.7695007
1: -19.2477093, 66.8586197, -18.0245533, 62.6214600, -81.8691711, 84.8831635
2: -10.5548000, 62.1071167, -9.8561020, 58.1786079, -68.7334061, 71.9632111
3: -14.7066040, 91.5520782, -13.7885504, 85.7612534, -100.4678574, 105.3406219
4: -20.0339489, 75.7693481, -18.6663570, 70.9539566, -90.9878998, 94.4357071

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -29.1225548, 104.1819839, -29.1225548, 104.1819839, -133.3045349, 133.3045349
1: -18.0245533, 62.6214600, -18.0245533, 62.6214600, -80.6460114, 80.6460114
2: -9.8561020, 58.1786079, -9.8561020, 58.1786079, -68.0347137, 68.0347137
3: -13.7885504, 85.7612534, -13.7885504, 85.7612534, -99.5498047, 99.5498047
4: -18.6663570, 70.9539566, -18.6663570, 70.9539566, -89.6203156, 89.6203156

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -30.8583736, 110.6469421, -31.1861115, 112.5543289, -143.4126892, 141.8330383
1: -19.2477093, 66.8586197, -19.3901558, 67.7447968, -86.9925079, 86.2487717
2: -10.5548000, 62.1071167, -10.5732336, 63.0687218, -73.6235199, 72.6803513
3: -14.7066040, 91.5520782, -14.7638540, 92.6199570, -107.3265610, 106.3159332
4: -20.0339489, 75.7693481, -20.1120605, 76.7380905, -96.7720413, 95.8814011

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 42

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -30.8583736, 110.6469421, -29.9327507, 107.8221512, -138.6805267, 140.5796814
1: -19.2477093, 66.8586197, -18.5010185, 64.6908798, -83.9385910, 85.3596115
2: -10.5548000, 62.1071167, -10.0674782, 60.2702751, -70.8250580, 72.1745834
3: -14.7066040, 91.5520782, -14.0960455, 88.4368286, -103.1434326, 105.6481094
4: -20.0339489, 75.7693481, -19.1031914, 73.2874603, -93.3213959, 94.8725281

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -29.1225548, 104.1819839, -31.1861115, 112.5543289, -141.6768799, 135.3681030
1: -18.0245533, 62.6214600, -19.3901558, 67.7447968, -85.7693481, 82.0116119
2: -9.8561020, 58.1786079, -10.5732336, 63.0687218, -72.9248199, 68.7518387
3: -13.7885504, 85.7612534, -14.7638540, 92.6199570, -106.4085083, 100.5251083
4: -18.6663570, 70.9539566, -20.1120605, 76.7380905, -95.4044495, 91.0660095

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -29.1225548, 104.1819839, -29.9327507, 107.8221512, -136.9447021, 134.1147308
1: -18.0245533, 62.6214600, -18.5010185, 64.6908798, -82.7154312, 81.1224594
2: -9.8561020, 58.1786079, -10.0674782, 60.2702751, -70.1263504, 68.2460861
3: -13.7885504, 85.7612534, -14.0960455, 88.4368286, -102.2253799, 99.8572998
4: -18.6663570, 70.9539566, -19.1031914, 73.2874603, -91.9538193, 90.0571365

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A1_B1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -22.2503166, 79.1481781, -27.1588058, 96.6940308, -118.9443512, 106.3069687
1: -13.5698557, 47.2135048, -16.6791821, 58.1066780, -71.6765366, 63.8926849
2: -7.3687849, 44.0084572, -9.1280499, 54.0097313, -61.3785095, 53.1365013
3: -10.5473537, 64.6359558, -12.8320627, 79.4705353, -90.0178757, 77.4680176
4: -13.8852253, 53.6403008, -17.2855644, 65.8328934, -79.7181168, 70.9258652

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8656425, upper bound: 147.8656425
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8656425, upper bound: 147.8791833
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -22.4488659, 80.2580490, -27.1588058, 96.6940308, -119.1428833, 107.4168396
1: -13.7406425, 48.0372925, -16.6791821, 58.1066780, -71.8473206, 64.7164688
2: -7.4545875, 44.8432198, -9.1280499, 54.0097313, -61.4643135, 53.9712677
3: -10.6910048, 65.7341690, -12.8320627, 79.4705353, -90.1615372, 78.5662308
4: -14.0259933, 54.5866356, -17.2855644, 65.8328934, -79.8588715, 71.8722000

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8656425, upper bound: 147.8656517
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8656425, upper bound: 147.8791833
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -30.4215851, 109.9125900, -23.2032948, 82.4177704, -112.8393555, 133.1158752
1: -18.8916531, 66.1009903, -14.2211962, 49.2841606, -68.1758118, 80.3221817
2: -10.2924824, 61.5798836, -7.7525816, 45.8768578, -56.1693344, 69.3324585
3: -14.3762436, 90.3226852, -11.0140276, 67.5324860, -81.9087296, 101.3367081
4: -19.5791988, 74.8753128, -14.5952921, 55.9940910, -75.5732803, 89.4705963

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8791952, upper bound: 147.8686143
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8791952, upper bound: 147.8690277
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -30.4215851, 109.9125900, -28.5931683, 102.3314209, -132.7530060, 138.5057526
1: -18.8916531, 66.1009903, -17.6705761, 61.6928253, -80.5844727, 83.7715607
2: -10.2924824, 61.5798836, -9.6752577, 57.3835068, -67.6759872, 71.2551346
3: -14.3762436, 90.3226852, -13.5521059, 84.3299179, -98.7061615, 103.8747940
4: -19.5791988, 74.8753128, -18.3629799, 69.9086761, -89.4878769, 93.2382812

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8768764, upper bound: 147.8742457
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8770136, upper bound: 147.8809260
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -22.2503166, 79.1481781, -26.3733807, 94.5028610, -116.7531738, 105.5215607
1: -13.5698557, 47.2135048, -16.2261848, 56.5625229, -70.1323776, 63.4396858
2: -7.3687849, 44.0084572, -8.8815937, 52.5701637, -59.9389420, 52.8900528
3: -10.5473537, 64.6359558, -12.5322237, 77.4584808, -88.0058212, 77.1681824
4: -13.8852253, 53.6403008, -16.7860546, 64.1028900, -77.9881134, 70.4263535

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8656517, upper bound: 147.8666755
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8656517, upper bound: 147.8799638
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -28.5218678, 103.2081528, -27.8759556, 100.1139450, -128.6358185, 131.0840759
1: -17.6144180, 61.7889404, -17.2764015, 60.1215057, -77.7359161, 79.0653381
2: -9.5762186, 57.5947495, -9.4690580, 55.8510780, -65.4272919, 67.0638046
3: -13.4369669, 84.3548965, -13.2886324, 82.3528824, -95.7898483, 97.6435242
4: -18.2046127, 69.9805679, -17.9292946, 68.1162033, -86.3208160, 87.9098511

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8791833, upper bound: 147.8684905
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8791833, upper bound: 147.8843113
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -22.4488659, 80.2580490, -26.3733807, 94.5028610, -116.9517136, 106.6314316
1: -13.7406425, 48.0372925, -16.2261848, 56.5625229, -70.3031616, 64.2634659
2: -7.4545875, 44.8432198, -8.8815937, 52.5701637, -60.0247459, 53.7248154
3: -10.6910048, 65.7341690, -12.5322237, 77.4584808, -88.1494827, 78.2663956
4: -14.0259933, 54.5866356, -16.7860546, 64.1028900, -78.1288834, 71.3726883

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8666837, upper bound: 147.8666837
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8656425, upper bound: 147.8797182
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -26.5584488, 95.2661819, -27.8759556, 100.1139450, -126.6723938, 123.1421356
1: -16.3612499, 57.3260651, -17.2764015, 60.1215057, -76.4827576, 74.6024628
2: -8.9254322, 53.3839149, -9.4690580, 55.8510780, -64.7765121, 62.8529739
3: -12.5899496, 78.3780975, -13.2886324, 82.3528824, -94.9428253, 91.6667328
4: -16.9151268, 64.9455032, -17.9292946, 68.1162033, -85.0313263, 82.8747864

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8799624, upper bound: 147.8697271
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8799624, upper bound: 147.8844207
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -23.4658470, 83.4173279, -25.4176807, 90.1015778, -113.5674210, 108.8350067
1: -14.4133177, 49.9048729, -15.7430477, 54.4464111, -68.8597260, 65.6479187
2: -7.8627238, 46.4424973, -8.6457529, 50.5347137, -58.3974380, 55.0882492
3: -11.1510811, 68.3978577, -12.1659594, 74.6058578, -85.7569427, 80.5638046
4: -14.8063288, 56.7009087, -16.3946724, 61.7497253, -76.5560532, 73.0955811

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B1_A1_A1_B1_B1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8640453, upper bound: 147.8750782
time: 0.75 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B1_B2

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8677871, upper bound: 147.8739706
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -23.4658470, 83.4173279, -26.7695236, 95.5459671, -119.0118103, 110.1868515
1: -14.4133177, 49.9048729, -16.7035770, 57.6046410, -72.0179596, 66.6084518
2: -7.8627238, 46.4424973, -9.1787004, 53.3813210, -61.2440453, 55.6211967
3: -11.1510811, 68.3978577, -12.8916302, 79.0410767, -90.1921539, 81.2894821
4: -14.8063288, 56.7009087, -17.4574413, 65.3044891, -80.1108170, 74.1583328

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A1_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8502581, upper bound: 147.8681225
time: 0.59 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_B1_A1_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8639240, upper bound: 147.8759589
time: 0.87 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B2_B2

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8591596, upper bound: 147.8656498
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -24.5587463, 88.0569992, -25.4176807, 90.1015778, -114.6603241, 113.4746780
1: -15.1891947, 52.6221390, -15.7430477, 54.4464111, -69.6355896, 68.3651886
2: -8.2939606, 48.9453201, -8.6457529, 50.5347137, -58.8286705, 57.5910721
3: -11.7343845, 72.1928177, -12.1659594, 74.6058578, -86.3402405, 84.3587723
4: -15.6375437, 59.7779083, -16.3946724, 61.7497253, -77.3872681, 76.1725769

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B1_A1_A2_B1_B1

### Relational analysis result of NS_A1_B2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8670999, upper bound: 147.8751477
time: 0.92 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2_B1_B2

### Relational analysis result of NS_A1_B2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8666755, upper bound: 147.8748315
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -24.5587463, 88.0569992, -26.7695236, 95.5459671, -120.1047134, 114.8265228
1: -15.1891947, 52.6221390, -16.7035770, 57.6046410, -72.7938309, 69.3257141
2: -8.2939606, 48.9453201, -9.1787004, 53.3813210, -61.6752777, 58.1240196
3: -11.7343845, 72.1928177, -12.8916302, 79.0410767, -90.7754593, 85.0844498
4: -15.6375437, 59.7779083, -17.4574413, 65.3044891, -80.9420242, 77.2353363

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8670999, upper bound: 147.8751477
time: 0.63 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2_B2_B2

### Relational analysis result of NS_A1_B2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8693371, upper bound: 147.8748315
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -28.8116131, 103.1824417, -25.4176807, 90.1015778, -118.9131851, 128.6001129
1: -17.8320045, 62.1673126, -15.7430477, 54.4464111, -72.2784119, 77.9103546
2: -9.7679558, 57.7995644, -8.6457529, 50.5347137, -60.3026695, 66.4453201
3: -13.6751442, 85.0107803, -12.1659594, 74.6058578, -88.2809982, 97.1767349
4: -18.5421829, 70.4488907, -16.3946724, 61.7497253, -80.2919083, 86.8435516

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8799724, upper bound: 147.8735394
time: 0.80 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8791959, upper bound: 147.8754092
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -26.9704113, 96.3104248, -25.4176807, 90.1015778, -117.0719833, 121.7281036
1: -16.6908493, 58.1834335, -15.7430477, 54.4464111, -71.1372604, 73.9264755
2: -9.1653557, 54.0866737, -8.6457529, 50.5347137, -59.7000656, 62.7324257
3: -12.8463850, 79.6591568, -12.1659594, 74.6058578, -87.4522324, 91.8251038
4: -17.3376427, 65.9333801, -16.3946724, 61.7497253, -79.0873718, 82.3280487

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8799724, upper bound: 147.8735394
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8791959, upper bound: 147.8754092
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -28.8116131, 103.1824417, -26.7695236, 95.5459671, -124.3575821, 129.9519501
1: -17.8320045, 62.1673126, -16.7035770, 57.6046410, -75.4366455, 78.8708878
2: -9.7679558, 57.7995644, -9.1787004, 53.3813210, -63.1492767, 66.9782562
3: -13.6751442, 85.0107803, -12.8916302, 79.0410767, -92.7162170, 97.9024124
4: -18.5421829, 70.4488907, -17.4574413, 65.3044891, -83.8466492, 87.9063187

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8620050, upper bound: 147.8672535
time: 0.57 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8588928, upper bound: 147.8655059
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -26.9912338, 96.3910828, -26.7695236, 95.5459671, -122.5372009, 123.1606064
1: -16.7048092, 58.2303963, -16.7035770, 57.6046410, -74.3094482, 74.9339752
2: -9.1734991, 54.1287651, -9.1787004, 53.3813210, -62.5548210, 63.3074532
3: -12.8568783, 79.7242203, -12.8916302, 79.0410767, -91.8979568, 92.6158447
4: -17.3538589, 65.9859009, -17.4574413, 65.3044891, -82.6583405, 83.4433212

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8620050, upper bound: 147.8754892
time: 0.83 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8588928, upper bound: 147.8689810
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -22.6421127, 80.2270584, -29.7535343, 105.4255371, -128.0676117, 109.9805908
1: -13.8473854, 48.0302620, -18.4677773, 64.1908264, -78.0382004, 66.4980392
2: -7.5458331, 44.7419701, -10.1829681, 59.6332970, -67.1791306, 54.9249382
3: -10.7480640, 65.7857132, -14.2208138, 87.7483139, -98.4963760, 80.0065308
4: -14.1836367, 54.5946350, -19.3269501, 72.7140732, -86.8977127, 73.9215851

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8692636, upper bound: 147.8841929
time: 0.57 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8692636, upper bound: 147.8841929
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -22.6117363, 80.2847519, -50.0817909, 185.6817932, -208.2935181, 130.3665466
1: -13.8171482, 47.9643135, -32.4277344, 108.3731766, -122.1903076, 80.3920364
2: -7.5254822, 44.6706352, -17.7074432, 99.2936478, -106.8191299, 62.3780785
3: -10.7262487, 65.6991196, -24.5080433, 150.2553253, -160.9815674, 90.2071457
4: -14.1555586, 54.5017052, -33.6131516, 122.7089005, -136.8644562, 88.1148529

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8586069, upper bound: 147.8732170
time: 0.84 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8586069, upper bound: 147.8897022
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -24.0256691, 85.9820557, -29.6048241, 104.9327545, -128.9584198, 115.5868835
1: -14.8283815, 51.4076080, -18.3804245, 63.9252777, -78.7536621, 69.7880173
2: -8.0925217, 47.8413010, -10.1364670, 59.3988533, -67.4913712, 57.9777641
3: -11.4807396, 70.5047302, -14.1579218, 87.3751907, -98.8559265, 84.6626511
4: -15.2435999, 58.4185181, -19.2369041, 72.4138031, -87.6574020, 77.6554260

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8735546, upper bound: 147.8875512
time: 0.87 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_B2

### Relational analysis result of NS_A1_B2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8735546, upper bound: 147.8875575
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -23.3332005, 83.4705582, -49.7460823, 184.7287903, -208.0619812, 133.2166138
1: -14.3403692, 49.8427963, -32.2250938, 107.6705017, -122.0108719, 82.0678864
2: -7.8160620, 46.4096107, -17.6038818, 98.6122437, -106.4283066, 64.0134888
3: -11.1230879, 68.3298035, -24.3723774, 149.2971649, -160.4202576, 92.7021790
4: -14.7142658, 56.6287231, -33.4376831, 121.8999939, -136.6142578, 90.0663986

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8714463, upper bound: 147.8880167
time: 0.83 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8696141, upper bound: 147.8793938
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -28.9132442, 103.5669708, -31.9577370, 114.5451126, -143.4583588, 135.5247040
1: -17.8996487, 62.4445076, -19.9770164, 69.2282104, -87.1278610, 82.4215240
2: -9.8057909, 58.0699806, -11.0028372, 64.1874008, -73.9931946, 69.0728149
3: -13.7253065, 85.3750687, -15.3477163, 94.7345428, -108.4598465, 100.7227859
4: -18.6112766, 70.7665939, -20.9270992, 78.3787079, -96.9899673, 91.6936951

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8620050, upper bound: 147.8678820
time: 0.92 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8567409, upper bound: 147.8578308
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -28.9132442, 103.5669708, -29.9214573, 106.6872711, -135.6005096, 133.4884338
1: -17.8996487, 62.4445076, -18.6861153, 64.6684418, -82.5680923, 81.1306152
2: -9.8057909, 58.0699806, -10.3091946, 59.9167862, -69.7225800, 68.3791656
3: -13.7253065, 85.3750687, -14.4094715, 88.5997391, -102.3250427, 99.7845383
4: -18.6112766, 70.7665939, -19.5935669, 73.2198563, -91.8311157, 90.3601608

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8620050, upper bound: 147.8678820
time: 0.57 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8567409, upper bound: 147.8655059
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -26.9912338, 96.3910828, -33.4729576, 119.9028015, -146.8939972, 129.8640442
1: -16.7048092, 58.2303963, -20.9950600, 72.6206284, -89.3254318, 79.2254562
2: -9.1734991, 54.1287651, -11.5750170, 67.3149948, -76.4884949, 65.7037811
3: -12.8568783, 79.7242203, -16.0926437, 99.4552231, -112.3121033, 95.8168640
4: -17.3538589, 65.9859009, -22.0184631, 82.2204437, -99.5743027, 88.0043564

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8804908, upper bound: 147.8859308
time: 1.12 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8773934, upper bound: 147.8906318
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -26.5051651, 94.5871887, -31.1186581, 111.2161942, -137.7213440, 125.7058411
1: -16.3650322, 57.0979042, -19.4000130, 67.2242889, -83.5893173, 76.4979172
2: -8.9809694, 53.0907135, -10.6623421, 62.3680344, -71.3490067, 63.7530556
3: -12.6095133, 78.1582642, -14.8629665, 92.0351181, -104.6446304, 93.0212250
4: -16.9784660, 64.7039871, -20.2521610, 76.1176910, -93.0961609, 84.9561462

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8805325, upper bound: 147.8801038
time: 0.83 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8805325, upper bound: 147.8801038
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -25.4176807, 90.1015778, -23.4658470, 83.4173279, -108.8350067, 113.5674210
1: -15.7430477, 54.4464111, -14.4133177, 49.9048729, -65.6479187, 68.8597260
2: -8.6457529, 50.5347137, -7.8627238, 46.4424973, -55.0882492, 58.3974342
3: -12.1659594, 74.6058578, -11.1510811, 68.3978577, -80.5638046, 85.7569427
4: -16.3946724, 61.7497253, -14.8063288, 56.7009087, -73.0955811, 76.5560532

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8750782, upper bound: 147.8640453
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A1_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8739706, upper bound: 147.8677871
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -26.7695236, 95.5459671, -23.4658470, 83.4173279, -110.1868515, 119.0118103
1: -16.7035770, 57.6046410, -14.4133177, 49.9048729, -66.6084518, 72.0179596
2: -9.1787004, 53.3813210, -7.8627238, 46.4424973, -55.6211967, 61.2440453
3: -12.8916302, 79.0410767, -11.1510811, 68.3978577, -81.2894821, 90.1921539
4: -17.4574413, 65.3044891, -14.8063288, 56.7009087, -74.1583252, 80.1108170

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8678113, upper bound: 147.8530760
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8759589, upper bound: 147.8639240
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8595349, upper bound: 147.8609765
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -25.4176807, 90.1015778, -24.5587463, 88.0569992, -113.4746780, 114.6603241
1: -15.7430477, 54.4464111, -15.1891947, 52.6221390, -68.3651886, 69.6355896
2: -8.6457529, 50.5347137, -8.2939606, 48.9453201, -57.5910721, 58.8286667
3: -12.1659594, 74.6058578, -11.7343845, 72.1928177, -84.3587646, 86.3402405
4: -16.3946724, 61.7497253, -15.6375437, 59.7779083, -76.1725769, 77.3872681

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8750782, upper bound: 147.8670999
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8739706, upper bound: 147.8693371
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -26.7695236, 95.5459671, -24.5587463, 88.0569992, -114.8265228, 120.1047134
1: -16.7035770, 57.6046410, -15.1891947, 52.6221390, -69.3257141, 72.7938309
2: -9.1787004, 53.3813210, -8.2939606, 48.9453201, -58.1240196, 61.6752777
3: -12.8916302, 79.0410767, -11.7343845, 72.1928177, -85.0844498, 90.7754593
4: -17.4574413, 65.3044891, -15.6375437, 59.7779083, -77.2353363, 80.9420242

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8750782, upper bound: 147.8650264
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8656425, upper bound: 147.8683019
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -25.4176807, 90.1015778, -28.8116131, 103.1824417, -128.6000977, 118.9131851
1: -15.7430477, 54.4464111, -17.8320045, 62.1673126, -77.9103546, 72.2784119
2: -8.6457529, 50.5347137, -9.7679558, 57.7995644, -66.4453201, 60.3026695
3: -12.1659594, 74.6058578, -13.6751442, 85.0107803, -97.1767349, 88.2809982
4: -16.3946724, 61.7497253, -18.5421829, 70.4488907, -86.8435516, 80.2919006

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8682255, upper bound: 147.8799724
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8754092, upper bound: 147.8791959
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -25.4176807, 90.1015778, -26.9704113, 96.3104248, -121.7281036, 117.0719757
1: -15.7430477, 54.4464111, -16.6908493, 58.1834335, -73.9264755, 71.1372604
2: -8.6457529, 50.5347137, -9.1653557, 54.0866737, -62.7324257, 59.7000694
3: -12.1659594, 74.6058578, -12.8463850, 79.6591568, -91.8251038, 87.4522324
4: -16.3946724, 61.7497253, -17.3376427, 65.9333801, -82.3280487, 79.0873718

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8664087, upper bound: 147.8858467
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8754092, upper bound: 147.8868585
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -26.7695236, 95.5459671, -28.8116131, 103.1824417, -129.9519501, 124.3575821
1: -16.7035770, 57.6046410, -17.8320045, 62.1673126, -78.8708878, 75.4366455
2: -9.1787004, 53.3813210, -9.7679558, 57.7995644, -66.9782562, 63.1492767
3: -12.8916302, 79.0410767, -13.6751442, 85.0107803, -97.9024124, 92.7162170
4: -17.4574413, 65.3044891, -18.5421829, 70.4488907, -87.9063187, 83.8466415

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8672535, upper bound: 147.8620050
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8655059, upper bound: 147.8609966
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -26.7695236, 95.5459671, -26.9912338, 96.3910828, -123.1606064, 122.5372009
1: -16.7035770, 57.6046410, -16.7048092, 58.2303963, -74.9339752, 74.3094482
2: -9.1787004, 53.3813210, -9.1734991, 54.1287651, -63.3074608, 62.5548210
3: -12.8916302, 79.0410767, -12.8568783, 79.7242203, -92.6158447, 91.8979568
4: -17.4574413, 65.3044891, -17.3538589, 65.9859009, -83.4433212, 82.6583405

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8672535, upper bound: 147.8677869
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8593314, upper bound: 147.8677922
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -27.2324657, 95.9196930, -24.7000523, 87.7459106, -114.9783783, 120.6197433
1: -16.7665958, 58.5133476, -15.2780151, 52.9823112, -69.7488937, 73.7913666
2: -9.2019768, 54.4350700, -8.3314219, 49.3616371, -58.5636101, 62.7664909
3: -12.9281368, 79.8853226, -11.7383986, 72.5906525, -85.5187912, 91.6237183
4: -17.4569378, 66.2755508, -15.7060623, 60.1965370, -77.6534576, 81.9816055

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8707479, upper bound: 147.8569107
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8707479, upper bound: 147.8734294
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -45.3171234, 168.8175964, -24.6928272, 87.9249496, -133.2420654, 193.5104218
1: -29.4450912, 97.4463654, -15.2675848, 52.9477348, -82.3928223, 112.7139511
2: -15.9968100, 88.9492950, -8.3210554, 49.3049660, -65.3017731, 97.2703476
3: -22.2136688, 135.5443420, -11.7297716, 72.5614624, -94.7751312, 147.2741089
4: -30.4155426, 110.2450256, -15.7004395, 60.1312065, -90.5467529, 125.9454575

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8739122, upper bound: 147.8591826
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8739122, upper bound: 147.8752254
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -33.6123505, 120.4252777, -30.5085926, 109.4097214, -143.0220642, 150.9338531
1: -21.0901127, 72.8470917, -19.0220032, 66.1805801, -87.2706680, 91.8690872
2: -11.6257772, 67.4980087, -10.4253616, 61.5210419, -73.1468201, 77.9233704
3: -16.1589336, 99.7994843, -14.5238323, 90.5798340, -106.7387543, 114.3233185
4: -22.1145821, 82.4763489, -19.7888851, 75.0032043, -97.1177826, 102.2652359

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -33.6123505, 120.4252777, -30.7524834, 111.0561600, -144.6685181, 151.1777649
1: -21.0901127, 72.8470917, -19.1110077, 66.8368073, -87.9269104, 91.9580994
2: -11.6257772, 67.4980087, -10.4146938, 62.2600479, -73.8858032, 77.9127045
3: -16.1589336, 99.7994843, -14.5355844, 91.3437271, -107.5026627, 114.3350677
4: -22.1145821, 82.4763489, -19.8127403, 75.7117081, -97.8262863, 102.2890854

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8977356, upper bound: 147.8870307
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8982490, upper bound: 147.8870261
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -21.7917290, 77.8041306, -26.6171818, 94.8257065, -116.6174240, 104.4212952
1: -13.4083481, 45.5996208, -16.4631348, 56.7545853, -70.1629333, 62.0627403
2: -7.3831434, 41.7376671, -9.0071535, 52.5779915, -59.9611282, 50.7448158
3: -10.3457680, 63.0119133, -12.5996399, 77.8969116, -88.2426758, 75.6115570
4: -14.0076847, 51.3448372, -17.0474644, 64.2688370, -78.2764969, 68.3923035

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A2_A1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8775123, upper bound: 147.8673188
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8752205, upper bound: 147.8655384
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8752422, upper bound: 147.8654754
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -30.9926300, 111.5501099, -30.2659416, 108.2507553, -139.2433777, 141.8160400
1: -19.3175087, 67.1986847, -18.8245525, 65.6497345, -84.9672394, 86.0232239
2: -10.5441504, 62.4327164, -10.3094978, 61.0737610, -71.6178818, 72.7421951
3: -14.7215710, 91.8939056, -14.3805285, 89.7970200, -104.5185928, 106.2744370
4: -20.0885963, 76.0336685, -19.5738239, 74.4004288, -94.4890137, 95.6074829

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A2_A2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8823138, upper bound: 147.8746025
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_A2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8861021, upper bound: 147.8850356
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -18.9916611, 66.8387070, -18.9252701, 66.6268311, -85.6184845, 85.7639771
1: -11.6649609, 38.9399948, -11.6234646, 38.7567062, -50.4216652, 50.5634575
2: -6.4185624, 35.4893227, -6.3978968, 35.3020897, -41.7206497, 41.8872108
3: -9.0818567, 54.1079369, -9.0532026, 53.8749390, -62.9567909, 63.1611404
4: -12.0993586, 43.9316406, -12.0571680, 43.7247124, -55.8240700, 55.9888077

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B1_A1_A1_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8873247, upper bound: 147.8825012
time: 0.58 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8797342, upper bound: 147.8797342
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -18.9916611, 66.8387070, -26.1570339, 92.6663208, -111.6579819, 92.9957352
1: -11.6649609, 38.9399948, -16.2293854, 56.1237526, -67.7887115, 55.1693764
2: -6.4185624, 35.4893227, -8.8749485, 52.1167908, -58.5353546, 44.3642578
3: -9.0818567, 54.1079369, -12.4578133, 76.8848419, -85.9666748, 66.5657501
4: -12.0993586, 43.9316406, -16.8578510, 63.6104660, -75.7098160, 60.7894821

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 15

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_B1_A1_A1_B2_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8873247, upper bound: 147.8864436
time: 0.85 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B2_A2

### Relational analysis result of NS_A2_B2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -147.8797342, upper bound: 147.8815742
time: 0.71 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.78 + 417.14 = 420.92 seconds
