## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 268.920435005728


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-97.3171921, 286.0284424, -97.3171921, 286.0284424, -383.3456421, 383.3456421)
1: (-69.2169495, 178.1541443, -69.2169495, 178.1541443, -247.3710938, 247.3710938)
2: (-76.4095612, 163.0122986, -76.4095612, 163.0122986, -239.4218445, 239.4218445)
3: (-68.0424042, 212.7014313, -68.0424042, 212.7014313, -280.7437744, 280.7437744)
4: (-111.7602081, 173.1322174, -111.7602081, 173.1322174, -284.8924255, 284.8924255)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.66 + 1.86 = 3.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -268.9365712, upper bound: 268.9365712

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -268.9325322, upper bound: 268.9161948
time: 0.95 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9199243, upper bound: 268.9199243
time: 0.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.76 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.76
Output dim: 4, lower bound: -268.9325322, upper bound: 268.9161948
NS_A2, status: Status.VERIFIED, split count: 1, time: 1.76
Output dim: 4, lower bound: -268.9199243, upper bound: 268.9199243

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -86.6519012, 255.5075226, -97.3171921, 286.0284424, -372.6803589, 352.8246765
1: -61.6857948, 159.5248108, -69.2169495, 178.1541443, -239.8399353, 228.7417603
2: -68.5178223, 146.0747070, -76.4095612, 163.0122986, -231.5301208, 222.4842529
3: -60.6795959, 190.2366943, -68.0424042, 212.7014313, -273.3810120, 258.2789917
4: -100.3572769, 154.6836090, -111.7602081, 173.1322174, -273.4895020, 266.4438171

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9161948, upper bound: 268.9161948
time: 0.66 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9161948, upper bound: 268.9161948
time: 0.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.97 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.97
Output dim: 4, lower bound: -268.9161948, upper bound: 268.9161948
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.97
Output dim: 4, lower bound: -268.9161948, upper bound: 268.9161948

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 3.52 + 4.73 = 8.25 seconds
