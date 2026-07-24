## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.045175422


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553)
1: (-0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822)
2: (-0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808)
3: (-0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506)
4: (-0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.67 + 0.75 = 1.42 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0465726, upper bound: 0.0465726

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159
time: 0.19 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.45 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 0.45
Output dim: 0, lower bound: -0.0450159, upper bound: 0.0450337
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 0.45
Output dim: 0, lower bound: -0.0450337, upper bound: 0.0450159

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.42 + 0.45 = 1.87 seconds
