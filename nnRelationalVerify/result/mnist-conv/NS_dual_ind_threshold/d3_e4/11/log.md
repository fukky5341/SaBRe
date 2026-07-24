## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.1665576092


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7390966, 2.7390966)
1: (-6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2134962, 2.2134960)
2: (8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2642365, 2.2642365)
3: (-6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9247456, 2.9247451)
4: (-11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9852571, 2.9852571)
5: (-13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.5011797, 2.5011792)
6: (-15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3238053, 2.3238053)
7: (-5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2520328, 3.2520332)
8: (-1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0681491, 2.0681491)
9: (-7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7144065, 2.7144065)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.67 + 35.91 = 58.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -1.1688954, upper bound: 1.1688953

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1648070, upper bound: 1.1577665
time: 14.45 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688675, upper bound: 1.1688662
time: 5.18 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 19.72 seconds
NS_A1, status: Status.VERIFIED, split count: 1, time: 19.72
Output dim: 2, lower bound: -1.1648070, upper bound: 1.1577665
NS_A2, status: Status.UNKNOWN, split count: 1, time: 19.72
Output dim: 2, lower bound: -1.1688675, upper bound: 1.1688662

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -9.0258846, -5.5622597, -9.0258932, -5.5622568, -2.7596607, 2.7251792
1: -6.5765324, -3.9591105, -6.5765352, -3.9591038, -2.2075744, 2.2108514
2: 8.3243246, 10.9320049, 8.3243179, 10.9320288, -2.2642126, 2.2531314
3: -6.1232677, -2.8826237, -6.1232758, -2.8826151, -2.9204893, 2.9423370
4: -11.8333645, -7.9824491, -11.8333731, -7.9824457, -2.9800339, 2.9852414
5: -13.6636324, -10.1825533, -13.6636429, -10.1825514, -2.4983902, 2.5127039
6: -15.6556368, -12.3172045, -15.6556492, -12.3171968, -2.3277698, 2.3129454
7: -5.5686011, -2.0476842, -5.5686102, -2.0476756, -3.2501769, 3.2462888
8: -1.9611845, 0.3840852, -1.9611893, 0.3840871, -2.0765119, 2.0609412
9: -7.3109035, -4.0054445, -7.3109140, -4.0054426, -2.7002220, 2.7123060

Time for backsubstitution: 21.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1577642, upper bound: 1.1648064
time: 5.23 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1577642, upper bound: 1.1648071
time: 6.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 33.71 seconds
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 33.71
Output dim: 2, lower bound: -1.1577642, upper bound: 1.1648064
NS_A2_B2, status: Status.VERIFIED, split count: 2, time: 33.71
Output dim: 2, lower bound: -1.1577642, upper bound: 1.1648071

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 58.58 + 53.43 = 112.01 seconds
