## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 2027.3678997182642


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-796.8801270, 1389.0648193, -796.8801270, 1389.0648193, -2185.9448242, 2185.9448242)
1: (-722.3177490, 1251.5062256, -722.3177490, 1251.5062256, -1973.8239746, 1973.8239746)
2: (-632.7144775, 1318.1571045, -632.7144775, 1318.1571045, -1950.8714600, 1950.8714600)
3: (-972.6072998, 1298.1092529, -972.6072998, 1298.1092529, -2270.7165527, 2270.7165527)
4: (-767.1251831, 1415.1643066, -767.1251831, 1415.1643066, -2182.2895508, 2182.2895508)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.47 + 2.12 = 3.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2027.3881736, upper bound: 2027.3881736

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3743673, upper bound: 2027.3678324
time: 0.69 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3677290, upper bound: 2027.3677290
time: 0.89 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.72 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.72
Output dim: 0, lower bound: -2027.3743673, upper bound: 2027.3678324
NS_B2, status: Status.VERIFIED, split count: 1, time: 1.72
Output dim: 0, lower bound: -2027.3677290, upper bound: 2027.3677290

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -796.8801270, 1389.0648193, -795.3016357, 1386.2761230, -2183.1562500, 2184.3664551
1: -722.3177490, 1251.5062256, -720.8872681, 1249.0040283, -1971.3217773, 1972.3935547
2: -632.7144775, 1318.1571045, -631.4618530, 1315.5272217, -1948.2415771, 1949.6188965
3: -972.6072998, 1298.1092529, -970.6871948, 1295.5187988, -2268.1259766, 2268.7963867
4: -767.1251831, 1415.1643066, -765.6003418, 1412.3479004, -2179.4726562, 2180.7646484

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3664183, upper bound: 2027.3677038
time: 0.78 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3664062, upper bound: 2027.3664886
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.02 seconds
NS_B1_A1, status: Status.VERIFIED, split count: 2, time: 3.02
Output dim: 0, lower bound: -2027.3664183, upper bound: 2027.3677038
NS_B1_A2, status: Status.VERIFIED, split count: 2, time: 3.02
Output dim: 0, lower bound: -2027.3664062, upper bound: 2027.3664886

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 3.60 + 4.74 = 8.33 seconds
