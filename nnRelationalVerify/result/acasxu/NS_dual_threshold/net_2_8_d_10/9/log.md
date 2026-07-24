## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 1399.2956865315


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-822.8517456, 858.8505249, -822.8517456, 858.8505249, -1681.7022705, 1681.7022705)
1: (-86.4612045, 61.2885971, -86.4612045, 61.2885971, -147.7498016, 147.7498016)
2: (-142.6965027, 158.6089478, -142.6965027, 158.6089478, -301.3054504, 301.3054504)
3: (-159.5850067, 101.1168900, -159.5850067, 101.1168900, -260.7019043, 260.7019043)
4: (-122.8630981, 130.1711121, -122.8630981, 130.1711121, -253.0341949, 253.0341797)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.76 + 1.72 = 4.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1420.6047579, upper bound: 1420.6047579

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.7849249, upper bound: 1420.0041576
time: 0.54 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.7244400, upper bound: 1419.7244400
time: 0.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.31 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.31
Output dim: 0, lower bound: -1419.7849249, upper bound: 1420.0041576
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.31
Output dim: 0, lower bound: -1419.7244400, upper bound: 1419.7244400

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -734.4067993, 770.0239868, -801.7481079, 837.9047852, -1572.3110352, 1571.7720947
1: -77.2703552, 54.6831322, -84.3157501, 59.7283554, -136.9987183, 138.9988861
2: -126.7010803, 142.2896271, -138.9044952, 154.7225189, -281.4235840, 281.1941223
3: -141.5429230, 90.4472046, -155.2444153, 98.5853806, -240.1282959, 245.6916199
4: -109.9565659, 116.6184921, -119.7274857, 126.9966278, -236.9531708, 236.3459778

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.7244400, upper bound: 1419.7244400
time: 0.52 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.7244400, upper bound: 1419.7244400
time: 0.56 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -736.0734863, 768.8969116, -791.1158447, 827.9284668, -1564.0019531, 1560.0125732
1: -77.2160339, 54.6596146, -83.2818527, 58.9302597, -136.1463013, 137.9414520
2: -127.0059280, 142.0561676, -136.9233093, 152.9064636, -279.9123840, 278.9794922
3: -142.1698456, 90.4003372, -152.9518127, 97.3847351, -239.5545654, 243.3521423
4: -110.5172577, 116.5389328, -118.1258926, 125.5117569, -236.0290070, 234.6648254

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.7244400, upper bound: 1419.7244400
time: 0.58 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.7244400, upper bound: 1419.7244400
time: 0.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.92 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.92
Output dim: 0, lower bound: -1419.7244400, upper bound: 1419.7244400
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.92
Output dim: 0, lower bound: -1419.7244400, upper bound: 1419.7244400
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.92
Output dim: 0, lower bound: -1419.7244400, upper bound: 1419.7244400
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.92
Output dim: 0, lower bound: -1419.7244400, upper bound: 1419.7244400

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -734.4067993, 770.0239868, -734.4067993, 770.0239868, -1504.4307861, 1504.4307861
1: -77.2703552, 54.6831322, -77.2703552, 54.6831322, -131.9534760, 131.9534912
2: -126.7010803, 142.2896271, -126.7010803, 142.2896271, -268.9907227, 268.9907227
3: -141.5429230, 90.4472046, -141.5429230, 90.4472046, -231.9901276, 231.9901123
4: -109.9565659, 116.6184921, -109.9565659, 116.6184921, -226.5750580, 226.5750580

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1415.9463444, upper bound: 1418.5908016
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.7849249, upper bound: 1420.0041576
time: 0.54 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -734.4067993, 770.0239868, -736.0734863, 768.8969116, -1503.3035889, 1506.0974121
1: -77.2703552, 54.6831322, -77.2160339, 54.6596146, -131.9299622, 131.8991699
2: -126.7010803, 142.2896271, -127.0059280, 142.0561676, -268.7572632, 269.2955627
3: -141.5429230, 90.4472046, -142.1698456, 90.4003372, -231.9432526, 232.6170349
4: -109.9565659, 116.6184921, -110.5172577, 116.5389328, -226.4954987, 227.1357422

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.6551701
time: 0.54 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5672514, upper bound: 1419.7357410
time: 0.51 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -736.0734863, 768.8969116, -734.4067993, 770.0239868, -1506.0974121, 1503.3035889
1: -77.2160339, 54.6596146, -77.2703552, 54.6831322, -131.8991699, 131.9299622
2: -127.0059280, 142.0561676, -126.7010803, 142.2896271, -269.2955627, 268.7572632
3: -142.1698456, 90.4003372, -141.5429230, 90.4472046, -232.6170349, 231.9432526
4: -110.5172577, 116.5389328, -109.9565659, 116.6184921, -227.1357422, 226.4954987

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4865597, upper bound: 1419.4097473
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5059947, upper bound: 1419.5059947
time: 0.50 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -736.0734863, 768.8969116, -736.0734863, 768.8969116, -1504.9704590, 1504.9704590
1: -77.2160339, 54.6596146, -77.2160339, 54.6596146, -131.8756409, 131.8756409
2: -127.0059280, 142.0561676, -127.0059280, 142.0561676, -269.0621033, 269.0621033
3: -142.1698456, 90.4003372, -142.1698456, 90.4003372, -232.5701904, 232.5701752
4: -110.5172577, 116.5389328, -110.5172577, 116.5389328, -227.0561829, 227.0561829

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4865597, upper bound: 1419.4097473
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5059947, upper bound: 1419.5059947
time: 0.54 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.93 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 0, lower bound: -1415.9463444, upper bound: 1418.5908016
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 0, lower bound: -1419.7849249, upper bound: 1420.0041576
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.6551701
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 0, lower bound: -1419.5672514, upper bound: 1419.7357410
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 0, lower bound: -1419.4865597, upper bound: 1419.4097473
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 0, lower bound: -1419.5059947, upper bound: 1419.5059947
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 0, lower bound: -1419.4865597, upper bound: 1419.4097473
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 0, lower bound: -1419.5059947, upper bound: 1419.5059947

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -765.3402710, 804.1170044, -728.5620117, 764.1286621, -1529.4686279, 1532.6789551
1: -80.5660553, 56.3464546, -76.6508636, 54.1991463, -134.7651825, 132.9973145
2: -132.2884521, 148.6591339, -125.6692581, 141.2051544, -273.4935913, 274.3283997
3: -148.3129578, 94.1042938, -140.4314117, 89.7038803, -238.0168152, 234.5356903
4: -115.1824951, 122.1069107, -109.1660767, 115.7158356, -230.8983154, 231.2729797

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.4347847, upper bound: 1418.8314353
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.8151832, upper bound: 1419.8151832
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.8151832, upper bound: 1419.8491479
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -730.2608643, 765.3975830, -734.4067993, 770.0239868, -1500.2849121, 1499.8043213
1: -76.8002930, 54.3216705, -77.2703552, 54.6831322, -131.4834290, 131.5920105
2: -125.9783020, 141.4437408, -126.7010803, 142.2896271, -268.2679443, 268.1448364
3: -140.8262482, 89.8896103, -141.5429230, 90.4472046, -231.2734375, 231.4325256
4: -109.4604721, 115.9188309, -109.9565659, 116.6184921, -226.0789642, 225.8753967

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4988421, upper bound: 1418.4374867
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.0777575, upper bound: 1418.0777575
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -675.3917847, 715.7163086, -564.7713623, 604.0980225, -1279.4897461, 1280.4876709
1: -71.5286407, 50.3708267, -60.3815727, 42.1991348, -113.7277756, 110.7523956
2: -116.5720139, 131.8995056, -97.5791168, 110.8747330, -227.4467468, 229.4786224
3: -129.8391571, 83.6788635, -108.5598602, 70.4000397, -200.2391510, 192.2387238
4: -100.8118362, 107.9824829, -84.4858322, 90.7173004, -191.5291138, 192.4683228

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.6551701
time: 0.56 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.6551701
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -734.4067993, 770.0239868, -707.9089355, 734.4812012, -1468.8878174, 1477.9328613
1: -77.2703552, 54.6831322, -73.7975998, 52.2891502, -129.5594940, 128.4807281
2: -126.7010803, 142.2896271, -122.4369431, 135.8135529, -262.5146179, 264.7265625
3: -141.5429230, 90.4472046, -137.8737946, 86.4741592, -228.0170898, 228.3209991
4: -109.9565659, 116.6184921, -107.4764862, 111.1459351, -221.1024628, 224.0949707

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5672514, upper bound: 1419.7357410
time: 0.63 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5672514, upper bound: 1419.7357410
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -564.7713623, 604.0980225, -675.3917847, 715.7163086, -1280.4876709, 1279.4897461
1: -60.3815727, 42.1991348, -71.5286407, 50.3708267, -110.7523956, 113.7277756
2: -97.5791168, 110.8747330, -116.5720139, 131.8995056, -229.4786224, 227.4467468
3: -108.5598602, 70.4000397, -129.8391571, 83.6788635, -192.2387238, 200.2391663
4: -84.4858322, 90.7173004, -100.8118362, 107.9824829, -192.4683228, 191.5291138

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.6551701, upper bound: 1419.4530403
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.6551701, upper bound: 1419.4530403
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -707.9089355, 734.4812012, -734.4067993, 770.0239868, -1477.9328613, 1468.8876953
1: -73.7975998, 52.2891502, -77.2703552, 54.6831322, -128.4807281, 129.5595093
2: -122.4369431, 135.8135529, -126.7010803, 142.2896271, -264.7265625, 262.5146484
3: -137.8737946, 86.4741592, -141.5429230, 90.4472046, -228.3209991, 228.0170898
4: -107.4764862, 111.1459351, -109.9565659, 116.6184921, -224.0949707, 221.1024628

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.7357410, upper bound: 1419.5672514
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.7357410, upper bound: 1419.5672514
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -564.7713623, 604.0980225, -677.5445557, 714.9533691, -1279.7247314, 1281.6425781
1: -60.3815727, 42.1991348, -71.6675797, 50.6162033, -110.9977722, 113.8667145
2: -97.5791168, 110.8747330, -116.4881592, 131.6285400, -229.2076569, 227.3628845
3: -108.5598602, 70.4000397, -129.3370819, 83.8089066, -192.3687744, 199.7370911
4: -84.4858322, 90.7173004, -100.6688156, 107.9659576, -192.4517822, 191.3861084

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.1090659, upper bound: 1418.9782309
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3903123, upper bound: 1419.3903123
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3903123, upper bound: 1419.4097473
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -707.9089355, 734.4812012, -736.0734863, 768.8969116, -1476.8056641, 1470.5546875
1: -73.7975998, 52.2891502, -77.2160339, 54.6596146, -128.4572144, 129.5051880
2: -122.4369431, 135.8135529, -127.0059280, 142.0561676, -264.4931030, 262.8194885
3: -137.8737946, 86.4741592, -142.1698456, 90.4003372, -228.2741394, 228.6440125
4: -107.4764862, 111.1459351, -110.5172577, 116.5389328, -224.0154114, 221.6631622

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4097473, upper bound: 1419.4865597
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4097473, upper bound: 1419.5059947
time: 0.51 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.97 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.97
Output dim: 0, lower bound: -1419.8151832, upper bound: 1419.8151832
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.97
Output dim: 0, lower bound: -1419.8151832, upper bound: 1419.8491479
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.97
Output dim: 0, lower bound: -1419.4988421, upper bound: 1418.4374867
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.97
Output dim: 0, lower bound: -1418.0777575, upper bound: 1418.0777575
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.97
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.6551701
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.97
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.6551701
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.97
Output dim: 0, lower bound: -1419.5672514, upper bound: 1419.7357410
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.97
Output dim: 0, lower bound: -1419.5672514, upper bound: 1419.7357410
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.97
Output dim: 0, lower bound: -1419.6551701, upper bound: 1419.4530403
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.97
Output dim: 0, lower bound: -1419.6551701, upper bound: 1419.4530403
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.97
Output dim: 0, lower bound: -1419.7357410, upper bound: 1419.5672514
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.97
Output dim: 0, lower bound: -1419.7357410, upper bound: 1419.5672514
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.97
Output dim: 0, lower bound: -1419.3903123, upper bound: 1419.3903123
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.97
Output dim: 0, lower bound: -1419.3903123, upper bound: 1419.4097473
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.97
Output dim: 0, lower bound: -1419.4097473, upper bound: 1419.4865597
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.97
Output dim: 0, lower bound: -1419.4097473, upper bound: 1419.5059947

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -765.3402710, 804.1170044, -765.3402710, 804.1170044, -1569.4569092, 1569.4570312
1: -80.5660553, 56.3464546, -80.5660553, 56.3464546, -136.9125061, 136.9125061
2: -132.2884521, 148.6591339, -132.2884521, 148.6591339, -280.9475708, 280.9475708
3: -148.3129578, 94.1042938, -148.3129578, 94.1042938, -242.4172363, 242.4172516
4: -115.1824951, 122.1069107, -115.1824951, 122.1069107, -237.2893829, 237.2893829

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -765.3402710, 804.1170044, -730.2608643, 765.3975830, -1530.7377930, 1534.3776855
1: -80.5660553, 56.3464546, -76.8002930, 54.3216705, -134.8877106, 133.1467438
2: -132.2884521, 148.6591339, -125.9783020, 141.4437408, -273.7321777, 274.6374512
3: -148.3129578, 94.1042938, -140.8262482, 89.8896103, -238.2025757, 234.9305267
4: -115.1824951, 122.1069107, -109.4604721, 115.9188309, -231.1013031, 231.5673828

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -712.4024048, 748.6575317, -642.9584351, 670.2901001, -1382.6923828, 1391.6156006
1: -75.0451050, 53.0047798, -67.3118134, 47.6303787, -122.6754837, 120.3165894
2: -122.9364471, 138.3192902, -111.3481293, 124.4404068, -247.3768311, 249.6674042
3: -137.5371552, 87.8071899, -126.9785919, 78.5557098, -216.0928650, 214.7857208
4: -106.8996887, 113.3577042, -98.7992172, 101.7323380, -208.6320190, 212.1568909

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2318949, upper bound: 1418.1707203
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2085088, upper bound: 1418.1657815
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -721.1214600, 755.9531250, -704.7446289, 739.3638916, -1460.4853516, 1460.6975098
1: -75.8224487, 53.5361786, -74.0917053, 52.1076775, -127.9301224, 127.6278839
2: -124.3463974, 139.7429504, -121.6444626, 136.7391205, -261.0855103, 261.3874207
3: -139.3504791, 88.6946487, -136.6948853, 86.5659561, -225.9164276, 225.3895264
4: -108.3246231, 114.5051575, -106.2793198, 112.0222321, -220.3468628, 220.7844849

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.0777575, upper bound: 1418.0777575
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.0777575, upper bound: 1418.0777575
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -579.8578491, 625.9525757, -564.7713623, 604.0980225, -1183.9558105, 1190.7238770
1: -62.3087692, 43.2692947, -60.3815727, 42.1991348, -104.5079041, 103.6508636
2: -100.5530777, 114.8444901, -97.5791168, 110.8747330, -211.4278107, 212.4236145
3: -112.2701035, 72.7176819, -108.5598602, 70.4000397, -182.6701050, 181.2775421
4: -87.1536713, 93.8437653, -84.4858322, 90.7173004, -177.8709717, 178.3295898

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5323729
time: 0.55 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4376145, upper bound: 1419.5850885
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -705.4064331, 733.4470825, -564.7713623, 604.0980225, -1309.5043945, 1298.2185059
1: -73.6685257, 52.2643814, -60.3815727, 42.1991348, -115.8676605, 112.6459503
2: -121.7820053, 135.6603851, -97.5791168, 110.8747330, -232.6567383, 233.2395020
3: -137.0787659, 86.2315521, -108.5598602, 70.4000397, -207.4787750, 194.7914124
4: -106.8931808, 110.8853073, -84.4858322, 90.7173004, -197.6104736, 195.3711395

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1416.4253702, upper bound: 1415.4805636
time: 0.60 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1414.0917687, upper bound: 1414.8921488
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -579.8578491, 625.9525757, -707.9089355, 734.4812012, -1314.3389893, 1333.8613281
1: -62.3087692, 43.2692947, -73.7975998, 52.2891502, -114.5979080, 117.0668869
2: -100.5530777, 114.8444901, -122.4369431, 135.8135529, -236.3666382, 237.2814331
3: -112.2701035, 72.7176819, -137.8737946, 86.4741592, -198.7442627, 210.5914764
4: -87.1536713, 93.8437653, -107.4764862, 111.1459351, -198.2996063, 201.3202515

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.7357410
time: 0.60 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4376145, upper bound: 1419.6045234
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -706.8775024, 734.6987915, -707.9089355, 734.4812012, -1441.3585205, 1442.6075439
1: -73.8210144, 52.3740158, -73.7975998, 52.2891502, -126.1101685, 126.1716080
2: -122.0527954, 135.8987885, -122.4369431, 135.8135529, -257.8663330, 258.3357239
3: -137.3982697, 86.4154739, -137.8737946, 86.4741592, -223.8724365, 224.2892761
4: -107.1148148, 111.0842209, -107.4764862, 111.1459351, -218.2607269, 218.5606995

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2417568, upper bound: 1419.5824390
time: 0.55 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4376145, upper bound: 1419.6545499
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -564.7713623, 604.0980225, -579.8578491, 625.9525757, -1190.7238770, 1183.9558105
1: -60.3815727, 42.1991348, -62.3087692, 43.2692947, -103.6508636, 104.5079041
2: -97.5791168, 110.8747330, -100.5530777, 114.8444901, -212.4236145, 211.4278107
3: -108.5598602, 70.4000397, -112.2701035, 72.7176819, -181.2775421, 182.6701050
4: -84.4858322, 90.7173004, -87.1536713, 93.8437653, -178.3295898, 177.8709717

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5323729, upper bound: 1419.2590854
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5850885, upper bound: 1419.4376145
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -564.7713623, 604.0980225, -705.4064331, 733.4470825, -1298.2185059, 1309.5043945
1: -60.3815727, 42.1991348, -73.6685257, 52.2643814, -112.6459503, 115.8676605
2: -97.5791168, 110.8747330, -121.7820053, 135.6603851, -233.2395020, 232.6567383
3: -108.5598602, 70.4000397, -137.0787659, 86.2315521, -194.7914124, 207.4787750
4: -84.4858322, 90.7173004, -106.8931808, 110.8853073, -195.3711395, 197.6104736

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1415.4805636, upper bound: 1416.4253702
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1414.8921488, upper bound: 1414.0917687
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -707.9089355, 734.4812012, -579.8578491, 625.9525757, -1333.8613281, 1314.3391113
1: -73.7975998, 52.2891502, -62.3087692, 43.2692947, -117.0668869, 114.5979156
2: -122.4369431, 135.8135529, -100.5530777, 114.8444901, -237.2814331, 236.3666382
3: -137.8737946, 86.4741592, -112.2701035, 72.7176819, -210.5914764, 198.7442627
4: -107.4764862, 111.1459351, -87.1536713, 93.8437653, -201.3202515, 198.2996063

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2567767, upper bound: 1418.2840803
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.7357410, upper bound: 1419.5672514
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.6045235, upper bound: 1419.5338618
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -707.9089355, 734.4812012, -706.8775024, 734.6987915, -1442.6075439, 1441.3585205
1: -73.7975998, 52.2891502, -73.8210144, 52.3740158, -126.1716080, 126.1101685
2: -122.4369431, 135.8135529, -122.0527954, 135.8987885, -258.3357239, 257.8663330
3: -137.8737946, 86.4741592, -137.3982697, 86.4154739, -224.2892761, 223.8724365
4: -107.4764862, 111.1459351, -107.1148148, 111.0842209, -218.5606995, 218.2607117

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2567768, upper bound: 1418.2840807
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5243418, upper bound: 1419.2617462
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.6045235, upper bound: 1419.5481088
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -564.7713623, 604.0980225, -564.7713623, 604.0980225, -1168.8693848, 1168.8693848
1: -60.3815727, 42.1991348, -60.3815727, 42.1991348, -102.5807037, 102.5807037
2: -97.5791168, 110.8747330, -97.5791168, 110.8747330, -208.4538574, 208.4538574
3: -108.5598602, 70.4000397, -108.5598602, 70.4000397, -178.9598999, 178.9598999
4: -84.4858322, 90.7173004, -84.4858322, 90.7173004, -175.2031250, 175.2031250

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1415.2612468, upper bound: 1414.1870562
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1413.7908828, upper bound: 1413.7908828
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -564.7713623, 604.0980225, -707.6264648, 734.2279663, -1298.9992676, 1311.7244873
1: -60.3815727, 42.1991348, -73.7577591, 52.2636642, -112.6452332, 115.9568939
2: -97.5791168, 110.8747330, -122.3777084, 135.7620850, -233.3412018, 233.2524414
3: -108.5598602, 70.4000397, -137.7992706, 86.4359741, -194.9958344, 208.1992950
4: -84.4858322, 90.7173004, -107.4236984, 111.1034622, -195.5892944, 198.1409912

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3375966, upper bound: 1419.2307376
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.3903123, upper bound: 1419.4097473
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -707.9089355, 734.4812012, -564.7713623, 604.0980225, -1312.0069580, 1299.2525635
1: -73.7975998, 52.2891502, -60.3815727, 42.1991348, -115.9967346, 112.6707230
2: -122.4369431, 135.8135529, -97.5791168, 110.8747330, -233.3116760, 233.3926697
3: -137.8737946, 86.4741592, -108.5598602, 70.4000397, -208.2738342, 195.0340271
4: -107.4764862, 111.1459351, -84.4858322, 90.7173004, -198.1937866, 195.6317749

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.0291713, upper bound: 1418.2302010
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7099686, upper bound: 1418.1030983
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.0069890, upper bound: 1419.0703680
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -707.9089355, 734.4812012, -707.9089355, 734.4812012, -1442.3898926, 1442.3900146
1: -73.7975998, 52.2891502, -73.7975998, 52.2891502, -126.0867310, 126.0867462
2: -122.4369431, 135.8135529, -122.4369431, 135.8135529, -258.2504883, 258.2504883
3: -137.8737946, 86.4741592, -137.8737946, 86.4741592, -224.3479614, 224.3479614
4: -107.4764862, 111.1459351, -107.4764862, 111.1459351, -218.6224213, 218.6224213

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7099686, upper bound: 1418.1206082
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.0069890, upper bound: 1419.0878778
time: 0.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.89 seconds
NS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.2318949, upper bound: 1418.1707203
NS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.2085088, upper bound: 1418.1657815
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1418.0777575, upper bound: 1418.0777575
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1418.0777575, upper bound: 1418.0777575
NS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5323729
NS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.4376145, upper bound: 1419.5850885
NS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1416.4253702, upper bound: 1415.4805636
NS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1414.0917687, upper bound: 1414.8921488
NS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.4530403, upper bound: 1419.7357410
NS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.4376145, upper bound: 1419.6045234
NS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.2417568, upper bound: 1419.5824390
NS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.4376145, upper bound: 1419.6545499
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.5323729, upper bound: 1419.2590854
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.5850885, upper bound: 1419.4376145
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1415.4805636, upper bound: 1416.4253702
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1414.8921488, upper bound: 1414.0917687
NS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.7357410, upper bound: 1419.5672514
NS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.6045235, upper bound: 1419.5338618
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.5243418, upper bound: 1419.2617462
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.6045235, upper bound: 1419.5481088
NS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1415.2612468, upper bound: 1414.1870562
NS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1413.7908828, upper bound: 1413.7908828
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.3375966, upper bound: 1419.2307376
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.3903123, upper bound: 1419.4097473
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1418.7099686, upper bound: 1418.1030983
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.0069890, upper bound: 1419.0703680
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1418.7099686, upper bound: 1418.1206082
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.89
Output dim: 0, lower bound: -1419.0069890, upper bound: 1419.0878778

## BFS NS instance: NS_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -656.1691284, 698.5343018, -513.8278809, 556.2310181, -1212.3997803, 1212.3621826
1: -69.7429123, 48.9311981, -55.5820045, 38.4726028, -108.2155151, 104.5131912
2: -113.1802139, 128.6649628, -89.3220139, 102.1858444, -215.3660583, 217.9869690
3: -126.0805969, 81.5323715, -100.5580215, 64.6113663, -190.6919556, 182.0903931
4: -97.9260559, 105.3565598, -78.2105560, 83.4764252, -181.4024658, 183.5671082

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2318949, upper bound: 1418.1707203
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1730072, upper bound: 1418.1592042
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -712.4024048, 748.6575317, -618.7394409, 637.8546753, -1350.2570801, 1367.3966064
1: -75.0451050, 53.0047798, -64.2126389, 45.5709953, -120.6161041, 117.2174225
2: -122.9364471, 138.3192902, -107.5342560, 118.6201935, -241.5566406, 245.8535461
3: -137.5371552, 87.8071899, -123.3448029, 74.8933716, -212.4305267, 211.1519928
4: -106.8996887, 113.3577042, -96.2585144, 96.6800537, -203.5797424, 209.6161957

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1402.6260025, upper bound: 1401.8012537
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1401.9236684, upper bound: 1397.1730307
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -639.2941284, 666.1225586, -704.7446289, 739.3638916, -1378.6577148, 1370.8669434
1: -66.8887939, 47.3324203, -74.0917053, 52.1076775, -118.9964752, 121.4241180
2: -110.7668762, 123.6957169, -121.6444626, 136.7391205, -247.5059662, 245.3401794
3: -126.3864975, 78.0549698, -136.6948853, 86.5659561, -212.9524231, 214.7498322
4: -98.3520813, 101.1142731, -106.2793198, 112.0222321, -210.3742981, 207.3935852

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1401.8012536, upper bound: 1396.9767257
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1395.1395605, upper bound: 1395.1395600
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -700.8231201, 734.8300781, -704.7446289, 739.3638916, -1440.1868896, 1439.5743408
1: -73.6290588, 51.7895660, -74.0917053, 52.1076775, -125.7367401, 125.8812561
2: -121.0087128, 135.9212799, -121.6444626, 136.7391205, -257.7478333, 257.5656738
3: -136.0563812, 86.0195312, -136.6948853, 86.5659561, -222.6223450, 222.7144012
4: -105.8010406, 111.3441391, -106.2793198, 112.0222321, -217.8232727, 217.6234589

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1396.9767263, upper bound: 1401.8012537
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1395.1395605, upper bound: 1395.1395600
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -568.5964355, 615.3504639, -520.5265503, 561.5841675, -1130.1805420, 1135.8769531
1: -61.2258987, 42.4158478, -55.9992790, 38.8239899, -100.0498886, 98.4151154
2: -98.6050262, 112.9004364, -89.5390854, 102.9940491, -201.5990143, 202.4395142
3: -110.0993652, 71.4179840, -99.4112320, 65.1949005, -175.2942657, 170.8291779
4: -85.5576782, 92.2700806, -77.8498535, 84.3572388, -169.9149170, 170.1199341

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5323729
time: 0.52 seconds

## Relational analysis of NS_A1_B2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5323729
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -568.9483032, 615.4647217, -701.0564575, 760.9191895, -1329.8673096, 1316.5208740
1: -61.2491798, 42.4668922, -75.6192703, 52.4171829, -113.6663666, 118.0861664
2: -98.6683960, 112.9437485, -121.8024521, 139.3765259, -238.0449219, 234.7461853
3: -110.2259598, 71.4415512, -135.5869446, 88.3582687, -198.5842285, 207.0285034
4: -85.6546021, 92.3021927, -103.7805634, 114.4485855, -200.1031342, 196.0827637

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4376145, upper bound: 1419.5850885
time: 0.49 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.4376145, upper bound: 1419.5850885
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -697.9319458, 724.6978149, -527.7058716, 562.8787842, -1260.8107910, 1252.4036865
1: -72.7597198, 51.6271858, -56.2904396, 39.4168282, -112.1765442, 107.9176254
2: -120.5503235, 134.0944366, -90.7378311, 103.3341751, -223.8844910, 224.8322754
3: -135.8184509, 85.1763458, -100.7727585, 65.6348801, -201.4533386, 185.9490967
4: -105.9616623, 109.5807266, -78.8829041, 84.5899048, -190.5515747, 188.4636230

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1414.0916935, upper bound: 1414.8921488
time: 0.62 seconds

## Relational analysis of NS_A1_B2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1414.0916935, upper bound: 1414.8921488
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -682.7901001, 708.2158813, -608.9118652, 657.7156982, -1340.5058594, 1317.1276855
1: -71.1065521, 50.3982201, -65.4957352, 45.6985741, -116.8051147, 115.8939514
2: -117.9438095, 131.0801697, -104.6482315, 120.5595703, -238.5033875, 235.7283783
3: -132.9464111, 83.2014618, -115.9146194, 76.4647217, -209.4111328, 199.1160736
4: -103.7695618, 107.1289597, -90.5495224, 98.8349304, -202.6044617, 197.6784515

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1414.0916935, upper bound: 1414.8921488
time: 0.58 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1414.0916935, upper bound: 1414.8921488
time: 0.54 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -535.2706299, 583.6050415, -699.6439209, 725.8706055, -1261.1411133, 1283.2487793
1: -57.9993362, 39.8736877, -72.9227676, 51.6079445, -109.6072845, 112.7964554
2: -92.6518936, 107.0407715, -120.9596939, 134.2189331, -226.8708191, 228.0004578
3: -103.4766312, 67.5666733, -136.1959229, 85.4260712, -188.9027100, 203.7625885
4: -80.6767960, 87.5416183, -106.2018967, 109.8545609, -190.5313263, 193.7435150

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2474992, upper bound: 1419.5243418
time: 0.55 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2474992, upper bound: 1419.6045235
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -735.2623291, 801.6467285, -695.4559326, 720.5531616, -1455.8153076, 1497.1022949
1: -79.4959869, 54.9512787, -72.4178162, 51.2014122, -130.6973877, 127.3690948
2: -127.7388916, 146.7774963, -120.3526001, 133.3501434, -261.0890503, 267.1300354
3: -142.0132141, 92.9911652, -135.7531586, 84.7861328, -226.7993317, 228.7442780
4: -108.7378311, 120.6224747, -105.8253632, 109.1242447, -217.8620758, 226.4478455

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2474992, upper bound: 1419.5243418
time: 0.59 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2474992, upper bound: 1419.6045235
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -696.3475342, 723.9306030, -677.2014160, 701.8816528, -1398.2292480, 1401.1319580
1: -72.6923065, 51.5223389, -70.4951859, 49.7640610, -122.4563675, 122.0175247
2: -120.2142105, 133.9175873, -117.0030289, 129.8107605, -250.0249634, 250.9205933
3: -135.3355103, 85.0701294, -131.7602234, 82.5221405, -217.8576355, 216.8303528
4: -105.5738373, 109.4654236, -102.8403702, 106.2690353, -211.8428497, 212.3057709

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2614613, upper bound: 1419.5824390
time: 1.17 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2614613, upper bound: 1419.5824390
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -693.8956909, 720.5190430, -790.8532104, 842.6197510, -1536.5153809, 1511.3723145
1: -72.3615799, 51.2704163, -84.3868332, 59.3563652, -131.7179413, 135.6572571
2: -119.9507141, 133.3665009, -136.9843750, 154.8421631, -274.7928772, 270.3508911
3: -135.2488098, 84.6701355, -152.6805420, 98.5492783, -233.7980804, 237.3506775
4: -105.4743958, 108.9894028, -117.5688400, 126.8253174, -232.2996979, 226.5582428

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5095799, upper bound: 1419.6545499
time: 0.59 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5095799, upper bound: 1419.6545499
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -520.5265503, 561.5841675, -568.5964355, 615.3504639, -1135.8769531, 1130.1804199
1: -55.9992790, 38.8239899, -61.2258987, 42.4158478, -98.4151154, 100.0498886
2: -89.5390854, 102.9940491, -98.6050262, 112.9004364, -202.4395142, 201.5990143
3: -99.4112320, 65.1949005, -110.0993652, 71.4179840, -170.8291779, 175.2942657
4: -77.8498535, 84.3572388, -85.5576782, 92.2700806, -170.1199341, 169.9149170

Time for backsubstitution: 2.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5323729, upper bound: 1419.2590854
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5323729, upper bound: 1419.2590854
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -701.0564575, 760.9191895, -568.9483032, 615.4647217, -1316.5208740, 1329.8674316
1: -75.6192703, 52.4171829, -61.2491798, 42.4668922, -118.0861588, 113.6663666
2: -121.8024521, 139.3765259, -98.6683960, 112.9437485, -234.7461853, 238.0449219
3: -135.5869446, 88.3582687, -110.2259598, 71.4415512, -207.0285034, 198.5842285
4: -103.7805634, 114.4485855, -85.6546021, 92.3021927, -196.0827637, 200.1031342

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5850885, upper bound: 1419.4376145
time: 0.55 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5850885, upper bound: 1419.4376145
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -527.7058716, 562.8787842, -697.9319458, 724.6978149, -1252.4036865, 1260.8107910
1: -56.2904396, 39.4168282, -72.7597198, 51.6271858, -107.9176254, 112.1765442
2: -90.7378311, 103.3341751, -120.5503235, 134.0944366, -224.8322754, 223.8844910
3: -100.7727585, 65.6348801, -135.8184509, 85.1763458, -185.9490967, 201.4533386
4: -78.8829041, 84.5899048, -105.9616623, 109.5807266, -188.4636230, 190.5515747

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1414.8921488, upper bound: 1414.0916935
time: 0.49 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1414.8921488, upper bound: 1414.0917687
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -608.9118652, 657.7156982, -682.7901001, 708.2158813, -1317.1276855, 1340.5058594
1: -65.4957352, 45.6985741, -71.1065521, 50.3982201, -115.8939438, 116.8051147
2: -104.6482315, 120.5595703, -117.9438095, 131.0801697, -235.7283783, 238.5033875
3: -115.9146194, 76.4647217, -132.9464111, 83.2014618, -199.1160736, 209.4111328
4: -90.5495224, 98.8349304, -103.7695618, 107.1289597, -197.6784515, 202.6044617

Time for backsubstitution: 2.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1414.8921488, upper bound: 1414.0916935
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1414.8921488, upper bound: 1414.0917687
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -699.6439209, 725.8706055, -535.2706299, 583.6050415, -1283.2487793, 1261.1411133
1: -72.9227676, 51.6079445, -57.9993362, 39.8736877, -112.7964554, 109.6072845
2: -120.9596939, 134.2189331, -92.6518936, 107.0407715, -228.0004578, 226.8708191
3: -136.1959229, 85.4260712, -103.4766312, 67.5666733, -203.7625885, 188.9027100
4: -106.2018967, 109.8545609, -80.6767960, 87.5416183, -193.7435150, 190.5313263

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5243418, upper bound: 1419.2474992
time: 0.54 seconds

## Relational analysis of NS_A2_B1_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5243418, upper bound: 1419.5338618
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -695.4559326, 720.5531616, -735.2623291, 801.6467285, -1497.1022949, 1455.8153076
1: -72.4178162, 51.2014122, -79.4959869, 54.9512787, -127.3690948, 130.6973877
2: -120.3526001, 133.3501434, -127.7388916, 146.7774963, -267.1300354, 261.0890503
3: -135.7531586, 84.7861328, -142.0132141, 92.9911652, -228.7442780, 226.7993317
4: -105.8253632, 109.1242447, -108.7378311, 120.6224747, -226.4478455, 217.8620758

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5243418, upper bound: 1419.2474992
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5243418, upper bound: 1419.5338618
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -677.2014160, 701.8816528, -696.3475342, 723.9306030, -1401.1319580, 1398.2292480
1: -70.4951859, 49.7640610, -72.6923065, 51.5223389, -122.0175247, 122.4563675
2: -117.0030289, 129.8107605, -120.2142105, 133.9175873, -250.9205933, 250.0249634
3: -131.7602234, 82.5221405, -135.3355103, 85.0701294, -216.8303528, 217.8576508
4: -102.8403702, 106.2690353, -105.5738373, 109.4654236, -212.3057709, 211.8428497

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5791436, upper bound: 1419.2617461
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.5791436, upper bound: 1419.2617461
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -790.8532104, 842.6197510, -693.8956909, 720.5190430, -1511.3723145, 1536.5153809
1: -84.3868332, 59.3563652, -72.3615799, 51.2704163, -135.6572571, 131.7179260
2: -136.9843750, 154.8421631, -119.9507141, 133.3665009, -270.3508911, 274.7928772
3: -152.6805420, 98.5492783, -135.2488098, 84.6701355, -237.3506775, 233.7980804
4: -117.5688400, 126.8253174, -105.4743958, 108.9894028, -226.5582428, 232.2996979

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.6626207, upper bound: 1419.5481088
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.6626207, upper bound: 1419.5481088
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -557.3410034, 595.9558716, -527.7058716, 562.8787842, -1120.2197266, 1123.6616211
1: -59.5782280, 41.6450615, -56.2904396, 39.4168282, -98.9950485, 97.9355011
2: -96.2369232, 109.4094238, -90.7378311, 103.3341751, -199.5710754, 200.1472473
3: -107.0481949, 69.4641495, -100.7727585, 65.6348801, -172.6830750, 170.2369080
4: -83.3951874, 89.5258560, -78.8829041, 84.5899048, -167.9850922, 168.4087524

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1413.7908828, upper bound: 1413.7908828
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1413.7908828, upper bound: 1413.7908828
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -545.0344238, 586.4285889, -608.9118652, 657.7156982, -1202.7501221, 1195.3403320
1: -58.5112801, 40.7605171, -65.4957352, 45.6985741, -104.2098541, 106.2562561
2: -93.9976959, 107.5757675, -104.6482315, 120.5595703, -214.5572662, 212.2239685
3: -104.4834213, 68.1934433, -115.9146194, 76.4647217, -180.9481506, 184.1080627
4: -81.6113739, 88.0556259, -90.5495224, 98.8349304, -180.4462891, 178.6051483

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1413.7908828, upper bound: 1413.7908828
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1413.7908828, upper bound: 1413.7908828
time: 0.54 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -520.5265503, 561.5841675, -699.4428711, 725.6860962, -1246.2124023, 1261.0268555
1: -55.9992790, 38.8239899, -72.8938141, 51.5894775, -107.5887527, 111.7178040
2: -89.5390854, 102.9940491, -120.9172363, 134.1816101, -223.7207031, 223.9112701
3: -99.4112320, 65.1949005, -136.1426239, 85.3982697, -184.8094330, 201.3374939
4: -77.8498535, 84.3572388, -106.1642380, 109.8237000, -187.6735382, 190.5214691

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1474814, upper bound: 1419.1510365
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1474814, upper bound: 1419.2307376
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -701.0564575, 760.9191895, -695.3459473, 720.4486084, -1421.5047607, 1456.2650146
1: -75.6192703, 52.4171829, -72.4015656, 51.1920586, -126.8113251, 124.8187485
2: -121.8024521, 139.3765259, -120.3312683, 133.3291779, -255.1316223, 259.7077942
3: -135.5869446, 88.3582687, -135.7276611, 84.7705383, -220.3574524, 224.0859375
4: -103.7805634, 114.4485855, -105.8076859, 109.1069183, -212.8874817, 220.2562256

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2001971, upper bound: 1419.3295656
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2001970, upper bound: 1419.4097473
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -721.5246582, 754.2047119, -552.0964966, 591.9736328, -1313.4982910, 1306.3012695
1: -75.5204849, 53.6202469, -59.1687660, 41.2950058, -116.8154907, 112.7890091
2: -124.7820663, 138.8873444, -95.3515854, 108.6439743, -233.4260254, 234.2389221
3: -139.8637848, 88.4701996, -106.1898041, 68.9352341, -208.7990112, 194.6600037
4: -109.0365601, 113.7427216, -82.8147125, 88.9070816, -197.9436340, 196.5574341

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.5307760, upper bound: 1418.0570864
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.7099686, upper bound: 1418.1030980
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -693.5084229, 720.4682007, -561.3460083, 601.0548706, -1294.5632324, 1281.8142090
1: -72.4024963, 51.1806183, -60.0667572, 41.9566612, -114.3591614, 111.2473755
2: -120.0330658, 133.2480927, -96.9875107, 110.3142319, -230.3472900, 230.2355957
3: -135.5054626, 84.7293167, -107.9114685, 70.0184097, -205.5238647, 192.6407166
4: -105.7276688, 109.0141068, -84.0188446, 90.2592697, -195.9869232, 193.0329437

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1415.8453944, upper bound: 1414.5132334
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1413.5400607, upper bound: 1413.8759020
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -721.5246582, 754.2047119, -697.8577271, 723.1194458, -1444.6440430, 1452.0625000
1: -75.5204849, 53.6202469, -72.6602936, 51.4229012, -126.9433746, 126.2805405
2: -124.7820663, 138.8873444, -120.7618256, 133.7819519, -258.5640259, 259.6491699
3: -139.8637848, 88.4701996, -136.2368317, 85.0944748, -224.9582520, 224.7070312
4: -109.0365601, 113.7427216, -106.2292404, 109.4614792, -218.4980469, 219.9719543

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8235875, upper bound: 1417.8235875
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8235875, upper bound: 1418.1206080
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -693.5084229, 720.4682007, -704.6269531, 731.2949829, -1424.8034668, 1425.0952148
1: -72.4024963, 51.1806183, -73.4731750, 52.0343742, -124.4368668, 124.6537933
2: -120.0330658, 133.2480927, -121.8848267, 135.2296600, -255.2627258, 255.1329193
3: -135.5054626, 84.7293167, -137.3245087, 86.0737305, -221.5791931, 222.0537872
4: -105.7276688, 109.0141068, -107.0702133, 110.6607666, -216.3884277, 216.0843048

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8235875, upper bound: 1417.8329215
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.8235875, upper bound: 1419.0878780
time: 0.64 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.27 seconds
NS_A1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.2318949, upper bound: 1418.1707203
NS_A1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.1730072, upper bound: 1418.1592042
NS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1402.6260025, upper bound: 1401.8012537
NS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1401.9236684, upper bound: 1397.1730307
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1401.8012536, upper bound: 1396.9767257
NS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.27
Output dim: 0, lower bound: -1395.1395605, upper bound: 1395.1395600
NS_A1_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1396.9767263, upper bound: 1401.8012537
NS_A1_B1_A2_B2_A2_A2, status: Status.VERIFIED, split count: 6, time: 4.27
Output dim: 0, lower bound: -1395.1395605, upper bound: 1395.1395600
NS_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5323729
NS_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.2590854, upper bound: 1419.5323729
NS_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.4376145, upper bound: 1419.5850885
NS_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.4376145, upper bound: 1419.5850885
NS_A1_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1414.0916935, upper bound: 1414.8921488
NS_A1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1414.0916935, upper bound: 1414.8921488
NS_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1414.0916935, upper bound: 1414.8921488
NS_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1414.0916935, upper bound: 1414.8921488
NS_A1_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.2474992, upper bound: 1419.5243418
NS_A1_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.2474992, upper bound: 1419.6045235
NS_A1_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.2474992, upper bound: 1419.5243418
NS_A1_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.2474992, upper bound: 1419.6045235
NS_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.2614613, upper bound: 1419.5824390
NS_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.2614613, upper bound: 1419.5824390
NS_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.5095799, upper bound: 1419.6545499
NS_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.5095799, upper bound: 1419.6545499
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.5323729, upper bound: 1419.2590854
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.5323729, upper bound: 1419.2590854
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.5850885, upper bound: 1419.4376145
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.5850885, upper bound: 1419.4376145
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1414.8921488, upper bound: 1414.0916935
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1414.8921488, upper bound: 1414.0917687
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1414.8921488, upper bound: 1414.0916935
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1414.8921488, upper bound: 1414.0917687
NS_A2_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.5243418, upper bound: 1419.2474992
NS_A2_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.5243418, upper bound: 1419.5338618
NS_A2_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.5243418, upper bound: 1419.2474992
NS_A2_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.5243418, upper bound: 1419.5338618
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.5791436, upper bound: 1419.2617461
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.5791436, upper bound: 1419.2617461
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.6626207, upper bound: 1419.5481088
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.6626207, upper bound: 1419.5481088
NS_A2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1413.7908828, upper bound: 1413.7908828
NS_A2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1413.7908828, upper bound: 1413.7908828
NS_A2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1413.7908828, upper bound: 1413.7908828
NS_A2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1413.7908828, upper bound: 1413.7908828
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.1474814, upper bound: 1419.1510365
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.1474814, upper bound: 1419.2307376
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.2001971, upper bound: 1419.3295656
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1419.2001970, upper bound: 1419.4097473
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1418.5307760, upper bound: 1418.0570864
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1418.7099686, upper bound: 1418.1030980
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1415.8453944, upper bound: 1414.5132334
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1413.5400607, upper bound: 1413.8759020
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1417.8235875, upper bound: 1417.8235875
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1417.8235875, upper bound: 1418.1206080
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1417.8235875, upper bound: 1417.8329215
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 0, lower bound: -1417.8235875, upper bound: 1419.0878780

## BFS NS instance: NS_A1_B1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -646.7606812, 689.4357300, -468.0868530, 510.4456177, -1157.2062988, 1157.5225830
1: -68.8059921, 48.2047691, -50.9124336, 34.9467773, -103.7527695, 99.1171875
2: -111.4951324, 126.9905701, -80.7417221, 93.8623505, -205.3574829, 207.7322540
3: -124.1655807, 80.4220505, -90.8899307, 59.0482903, -183.2138672, 171.3119659
4: -96.5176849, 104.0016708, -71.1053619, 76.6843262, -173.2020111, 175.1069946

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.2782944, upper bound: 1414.5806780
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.2782944, upper bound: 1418.1592042
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -642.4799194, 685.0081177, -634.1583252, 694.6873779, -1337.1672363, 1319.1665039
1: -68.3603287, 47.8546829, -69.0794907, 47.2607841, -115.6211090, 116.9341736
2: -110.6656494, 126.2213593, -110.9308701, 127.2729568, -237.9385986, 237.1522064
3: -123.3205795, 79.8814621, -124.7620392, 80.4648056, -203.7853851, 204.6434631
4: -95.9326477, 103.3703003, -95.2858047, 104.4914169, -200.4240723, 198.6560974

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.2782944, upper bound: 1414.5806780
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1417.2782944, upper bound: 1418.1592042
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -687.2393799, 720.4101562, -618.7394409, 637.8546753, -1325.0939941, 1339.1492920
1: -72.1712875, 50.9814873, -64.2126389, 45.5709953, -117.7422791, 115.1941223
2: -118.4070129, 133.2366028, -107.5342560, 118.6201935, -237.0272064, 240.7708588
3: -132.8134766, 84.4873123, -123.3448029, 74.8933716, -207.7068329, 207.8320923
4: -103.2925186, 109.1396790, -96.2585144, 96.6800537, -199.9725647, 205.3981934

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1396.8161565, upper bound: 1400.9034727
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1396.8125081, upper bound: 1400.9502468
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -674.8118896, 692.5987549, -596.9053955, 609.3739014, -1284.1856689, 1289.5039062
1: -69.9371796, 50.4151573, -61.4926186, 43.8814354, -113.8186188, 111.9077606
2: -116.1791229, 128.6074982, -103.9165497, 113.5997009, -229.7787781, 232.5240479
3: -131.4054718, 81.9473648, -119.6248169, 71.7295151, -203.1349640, 201.5721741
4: -102.5136185, 104.9131927, -93.4696884, 92.4644165, -194.9780121, 198.3828735

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1400.9393169, upper bound: 1396.3053933
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1395.0495517, upper bound: 1394.5073919
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -639.2941284, 666.1225586, -682.0796509, 713.7404785, -1353.0346680, 1348.2020264
1: -66.8887939, 47.3324203, -71.4865112, 50.3444252, -117.2332153, 118.8189316
2: -110.7668762, 123.6957169, -117.6716003, 132.1242371, -242.8910828, 241.3673096
3: -126.3864975, 78.0549698, -132.3785400, 83.5764771, -209.9629822, 210.4334869
4: -98.3520813, 101.1142731, -102.9716263, 108.1913376, -206.5433960, 204.0858765

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 3

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1400.9034725, upper bound: 1396.8161559
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1400.9502466, upper bound: 1396.8125074
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -678.3955688, 709.3696899, -704.7446289, 739.3638916, -1417.7592773, 1414.1141357
1: -71.0548935, 50.0403824, -74.0917053, 52.1076775, -123.1625671, 124.1320877
2: -117.0623779, 131.3419647, -121.6444626, 136.7391205, -253.8014832, 252.9864197
3: -131.7716827, 83.0514984, -136.6948853, 86.5659561, -218.3376465, 219.7463837
4: -102.5150604, 107.5438156, -106.2793198, 112.0222321, -214.5372925, 213.8231354

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1396.6907443, upper bound: 1401.3980493
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1396.4193587, upper bound: 1400.3978846
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -535.2706299, 583.6050415, -520.5265503, 561.5841675, -1096.8544922, 1104.1314697
1: -57.9993362, 39.8736877, -55.9992790, 38.8239899, -96.8233261, 95.8729630
2: -92.6518936, 107.0407715, -89.5390854, 102.9940491, -195.6459045, 196.5798645
3: -103.4766312, 67.5666733, -99.4112320, 65.1949005, -168.6715240, 166.9778595
4: -80.6767960, 87.5416183, -77.8498535, 84.3572388, -165.0340271, 165.3914642

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A1_B2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -735.2623291, 801.6467285, -520.5265503, 561.5841675, -1296.8463135, 1322.1729736
1: -79.4959869, 54.9512787, -55.9992790, 38.8239899, -118.3199768, 110.9505615
2: -127.7388916, 146.7774963, -89.5390854, 102.9940491, -230.7329102, 236.3165894
3: -142.0132141, 92.9911652, -99.4112320, 65.1949005, -207.2080994, 192.4023132
4: -108.7378311, 120.6224747, -77.8498535, 84.3572388, -193.0950623, 198.4722900

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -534.1478882, 582.4830933, -701.0564575, 760.9191895, -1295.0671387, 1283.5395508
1: -57.8855362, 39.7885323, -75.6192703, 52.4171829, -110.3027191, 115.4077988
2: -92.4440384, 106.8374863, -121.8024521, 139.3765259, -231.8205414, 228.6399384
3: -103.2631454, 67.4288559, -135.5869446, 88.3582687, -191.6214142, 203.0157623
4: -80.5232468, 87.3768082, -103.7805634, 114.4485855, -194.9717865, 191.1573792

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_B2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -735.2442627, 801.6259766, -701.0564575, 760.9191895, -1496.1634521, 1502.6820068
1: -79.4940948, 54.9498253, -75.6192703, 52.4171829, -131.9112854, 130.5690765
2: -127.7359772, 146.7740631, -121.8024521, 139.3765259, -267.1124878, 268.5764771
3: -142.0104523, 92.9887466, -135.5869446, 88.3582687, -230.3687134, 228.5756836
4: -108.7357025, 120.6195526, -103.7805634, 114.4485855, -223.1842194, 224.4001160

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

## BFS NS instance: NS_A1_B2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -669.7799683, 690.2387695, -527.7058716, 562.8787842, -1232.6586914, 1217.9445801
1: -69.3351669, 49.2334213, -56.2904396, 39.4168282, -108.7519989, 105.5238571
2: -115.7555466, 127.9645615, -90.7378311, 103.3341751, -219.0897064, 218.7023773
3: -130.9208679, 81.1337814, -100.7727585, 65.6348801, -196.5557556, 181.9065399
4: -102.2964172, 104.5212631, -78.8829041, 84.5899048, -186.8863220, 183.4041748

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -728.8052979, 756.3417969, -527.7058716, 562.8787842, -1291.6840820, 1284.0476074
1: -75.9114609, 53.6821136, -56.2904396, 39.4168282, -115.3282928, 109.9725494
2: -125.6333771, 139.9366302, -90.7378311, 103.3341751, -228.9675293, 230.6744690
3: -141.4862061, 88.8512344, -100.7727585, 65.6348801, -207.1210938, 189.6239929
4: -110.5063782, 114.5356064, -78.8829041, 84.5899048, -195.0962830, 193.4185181

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A1_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -669.7799683, 690.2387695, -608.9118652, 657.7156982, -1327.4956055, 1299.1506348
1: -69.3351669, 49.2334213, -65.4957352, 45.6985741, -115.0337296, 114.7291565
2: -115.7555466, 127.9645615, -104.6482315, 120.5595703, -236.3151245, 232.6127472
3: -130.9208679, 81.1337814, -115.9146194, 76.4647217, -207.3855896, 197.0484009
4: -102.2964172, 104.5212631, -90.5495224, 98.8349304, -201.1313477, 195.0707855

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -728.8052979, 756.3417969, -608.9118652, 657.7156982, -1386.5209961, 1365.2536621
1: -75.9114609, 53.6821136, -65.4957352, 45.6985741, -121.6100311, 119.1778488
2: -125.6333771, 139.9366302, -104.6482315, 120.5595703, -246.1929474, 244.5848389
3: -141.4862061, 88.8512344, -115.9146194, 76.4647217, -217.9509277, 204.7658539
4: -110.5063782, 114.5356064, -90.5495224, 98.8349304, -209.3412933, 205.0851135

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_B2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -535.2706299, 583.6050415, -677.2014160, 701.8816528, -1237.1523438, 1260.8063965
1: -57.9993362, 39.8736877, -70.4951859, 49.7640610, -107.7633972, 110.3688736
2: -92.6518936, 107.0407715, -117.0030289, 129.8107605, -222.4626465, 224.0437775
3: -103.4766312, 67.5666733, -131.7602234, 82.5221405, -185.9987793, 199.3269043
4: -80.6767960, 87.5416183, -102.8403702, 106.2690353, -186.9458160, 190.3819580

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2999859, upper bound: 1419.8098053
time: 0.59 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.2982189, upper bound: 1419.7970893
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -535.2706299, 583.6050415, -790.8532104, 842.6197510, -1377.8902588, 1374.4582520
1: -57.9993362, 39.8736877, -84.3868332, 59.3563652, -117.3556824, 124.2605209
2: -92.6518936, 107.0407715, -136.9843750, 154.8421631, -247.4940491, 244.0251007
3: -103.4766312, 67.5666733, -152.6805420, 98.5492783, -202.0259094, 220.2472076
4: -80.6767960, 87.5416183, -117.5688400, 126.8253174, -207.5021057, 205.1104584

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B2_A1_A1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.4723695, upper bound: 1419.3094427
time: 0.57 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1270318, upper bound: 1419.5849418
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -735.2623291, 801.6467285, -677.2014160, 701.8816528, -1437.1440430, 1478.8479004
1: -79.4959869, 54.9512787, -70.4951859, 49.7640610, -129.2600403, 125.4464645
2: -127.7388916, 146.7774963, -117.0030289, 129.8107605, -257.5496521, 263.7805176
3: -142.0132141, 92.9911652, -131.7602234, 82.5221405, -224.5353546, 224.7513733
4: -108.7378311, 120.6224747, -102.8403702, 106.2690353, -215.0068665, 223.4627838

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 8

## BFS NS instance: NS_A1_B2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -735.2623291, 801.6467285, -790.8532104, 842.6197510, -1577.8820801, 1592.5000000
1: -79.4959869, 54.9512787, -84.3868332, 59.3563652, -138.8523560, 139.3380890
2: -127.7388916, 146.7774963, -136.9843750, 154.8421631, -282.5810547, 283.7618713
3: -142.0132141, 92.9911652, -152.6805420, 98.5492783, -240.5624847, 245.6716614
4: -108.7378311, 120.6224747, -117.5688400, 126.8253174, -235.5631409, 238.1913147

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1418.4409290, upper bound: 1419.1779366
time: 0.56 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B2_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1419.1034694, upper bound: 1419.3974746
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -671.3620605, 697.9876099, -677.2014160, 701.8816528, -1373.2436523, 1375.1889648
1: -70.0229797, 49.4841576, -70.4951859, 49.7640610, -119.7870407, 119.9793396
2: -115.8662262, 129.1412964, -117.0030289, 129.8107605, -245.6769714, 246.1443024
3: -130.4348145, 81.9095840, -131.7602234, 82.5221405, -212.9569550, 213.6697845
4: -101.8773499, 105.5730362, -102.8403702, 106.2690353, -208.1463928, 208.4133453

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1401.5243097, upper bound: 1395.2466042
time: 0.60 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1400.2755975, upper bound: 1394.8741246
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -818.7238770, 875.0343018, -677.2014160, 701.8816528, -1520.6054688, 1552.2357178
1: -87.4468002, 61.3255463, -70.4951859, 49.7640610, -137.2108459, 131.8206940
2: -142.1504669, 160.8632965, -117.0030289, 129.8107605, -271.9611816, 277.8663025
3: -158.5834198, 102.2333298, -131.7602234, 82.5221405, -241.1055603, 233.9935608
4: -121.6414032, 131.8480530, -102.8403702, 106.2690353, -227.9104309, 234.6884155

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1401.5243097, upper bound: 1395.2466042
time: 0.59 seconds

## Relational analysis of NS_A1_B2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1400.2755975, upper bound: 1394.8741246
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -671.3620605, 697.9876099, -790.8532104, 842.6197510, -1513.9818115, 1488.8408203
1: -70.0229797, 49.4841576, -84.3868332, 59.3563652, -129.3793335, 133.8709869
2: -115.8662262, 129.1412964, -136.9843750, 154.8421631, -270.7083740, 266.1256714
3: -130.4348145, 81.9095840, -152.6805420, 98.5492783, -228.9841003, 234.5900879
4: -101.8773499, 105.5730362, -117.5688400, 126.8253174, -228.7026672, 223.1418610

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1394.9812086, upper bound: 1394.4375183
time: 0.59 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1394.4456616, upper bound: 1393.1562170
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -818.7904053, 875.0876465, -790.8532104, 842.6197510, -1661.4100342, 1665.9407959
1: -87.4532394, 61.3305168, -84.3868332, 59.3563652, -146.8096008, 145.7173462
2: -142.1619110, 160.8736572, -136.9843750, 154.8421631, -297.0040283, 297.8580322
3: -158.5969391, 102.2411880, -152.6805420, 98.5492783, -257.1462097, 254.9217224
4: -121.6506119, 131.8572388, -117.5688400, 126.8253174, -248.4758911, 249.4260864

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1401.4995273, upper bound: 1395.2416480
time: 0.66 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1394.4456616, upper bound: 1393.1562170
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -520.5265503, 561.5841675, -535.2706299, 583.6050415, -1104.1314697, 1096.8544922
1: -55.9992790, 38.8239899, -57.9993362, 39.8736877, -95.8729630, 96.8233261
2: -89.5390854, 102.9940491, -92.6518936, 107.0407715, -196.5798645, 195.6459045
3: -99.4112320, 65.1949005, -103.4766312, 67.5666733, -166.9778595, 168.6715240
4: -77.8498535, 84.3572388, -80.6767960, 87.5416183, -165.3914642, 165.0340271

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.48 + 415.88 = 420.36 seconds
