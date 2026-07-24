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
execution time: IAR + RelationalAnalysis = 1.61 + 1.84 = 3.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -268.9365712, upper bound: 268.9365712

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9199243, upper bound: 268.9161948
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -268.9161948, upper bound: 268.9199243
time: 0.61 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.82 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 1.82
Output dim: 4, lower bound: -268.9199243, upper bound: 268.9161948
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 1.82
Output dim: 4, lower bound: -268.9161948, upper bound: 268.9199243

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.45 + 1.82 = 5.27 seconds
