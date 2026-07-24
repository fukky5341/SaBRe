## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 2.7638016924


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331)
1: (-0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803)
2: (-1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662)
3: (-1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606)
4: (-1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.77 + 0.98 = 1.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.7804846, upper bound: 2.7804846

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804502
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804673
time: 0.28 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.65 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.65
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804502
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.65
Output dim: 0, lower bound: -2.7804502, upper bound: 2.7804673

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804290
time: 0.31 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7802298
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804438
time: 0.30 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.38 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.38
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7803401
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.38
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804290
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.38
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7802298
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.38
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7804438

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7800772
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7803401
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7772637
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7802298
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7803173
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7804438
time: 0.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.42 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.42
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7800772
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.42
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7803401
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.42
Output dim: 0, lower bound: -2.7802298, upper bound: 2.7772637
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.42
Output dim: 0, lower bound: -2.7801314, upper bound: 2.7804290
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.42
Output dim: 0, lower bound: -2.7804290, upper bound: 2.7801314
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.42
Output dim: 0, lower bound: -2.7772637, upper bound: 2.7802298
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.42
Output dim: 0, lower bound: -2.7803401, upper bound: 2.7803173
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.42
Output dim: 0, lower bound: -2.7800772, upper bound: 2.7804438

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7798526
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7771159
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800327
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799838, upper bound: 2.7800336
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7770397
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799926, upper bound: 2.7768811
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800654
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7800947
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7799066
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800654, upper bound: 2.7768811
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7799926
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7771057
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800336, upper bound: 2.7799838
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800327, upper bound: 2.7768811
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7801145
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7798526, upper bound: 2.7801145
time: 0.31 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.42 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.42
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7798526
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.42
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7771159
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.42
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800327
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.42
Output dim: 0, lower bound: -2.7799838, upper bound: 2.7800336
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.42
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7770397
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.42
Output dim: 0, lower bound: -2.7799926, upper bound: 2.7768811
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.42
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800654
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.42
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7800947
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.42
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7799066
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.42
Output dim: 0, lower bound: -2.7800654, upper bound: 2.7768811
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.42
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7799926
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.42
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7771057
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.42
Output dim: 0, lower bound: -2.7800336, upper bound: 2.7799838
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.42
Output dim: 0, lower bound: -2.7800327, upper bound: 2.7768811
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.42
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7801145
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.42
Output dim: 0, lower bound: -2.7798526, upper bound: 2.7801145

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7798526
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800401, upper bound: 2.7768811
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7771159
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800327
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800039
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799838, upper bound: 2.7800336
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7792388, upper bound: 2.7799277
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771057, upper bound: 2.7770397
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799926, upper bound: 2.7768811
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7795280, upper bound: 2.7768811
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800654
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800065
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7800947
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769265
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7769265, upper bound: 2.7768811
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800947, upper bound: 2.7799066
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800065, upper bound: 2.7768811
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800654, upper bound: 2.7768811
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7795280
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7799926
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770397, upper bound: 2.7771057
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7799277, upper bound: 2.7792388
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800336, upper bound: 2.7799838
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800039, upper bound: 2.7768811
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800327, upper bound: 2.7768811
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7801145
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800401
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7798526, upper bound: 2.7801145
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
time: 0.27 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.41 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7798526
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7800401, upper bound: 2.7768811
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7801145, upper bound: 2.7771159
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800327
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800039
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7799838, upper bound: 2.7800336
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7792388, upper bound: 2.7799277
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7771057, upper bound: 2.7770397
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7799926, upper bound: 2.7768811
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7795280, upper bound: 2.7768811
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800654
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800065
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7799066, upper bound: 2.7800947
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7769265
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7769265, upper bound: 2.7768811
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7800947, upper bound: 2.7799066
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7800065, upper bound: 2.7768811
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7800654, upper bound: 2.7768811
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7795280
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7799926
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7770397, upper bound: 2.7771057
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7799277, upper bound: 2.7792388
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7800336, upper bound: 2.7799838
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7800039, upper bound: 2.7768811
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7800327, upper bound: 2.7768811
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7801145
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7800401
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7798526, upper bound: 2.7801145
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.41
Output dim: 0, lower bound: -2.7768811, upper bound: 2.7768811

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788146
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7779866, upper bound: 2.7788146
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7752518, upper bound: 2.7746755
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782794, upper bound: 2.7746755
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782361, upper bound: 2.7746755
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754041, upper bound: 2.7790491
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7781960, upper bound: 2.7790491
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788344
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751457, upper bound: 2.7789023
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7755174
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750874, upper bound: 2.7753317
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7770375, upper bound: 2.7746755
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7786160, upper bound: 2.7746755
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747076, upper bound: 2.7746755
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7778045, upper bound: 2.7746755
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7783349
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753225
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782929
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753309
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7783621
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7753216
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746913
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747043
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747043, upper bound: 2.7746755
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746913, upper bound: 2.7746755
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753216, upper bound: 2.7787676
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783621, upper bound: 2.7787676
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753309, upper bound: 2.7746755
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7782929, upper bound: 2.7746755
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753225, upper bound: 2.7746755
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7783349, upper bound: 2.7746755
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7778045
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747076
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7786160
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7770375
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7753317, upper bound: 2.7750874
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755174, upper bound: 2.7746755
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789023, upper bound: 2.7751457
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788344, upper bound: 2.7746755
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7781960
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7754041
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782361
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782794
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7752518
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7779866
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7746755
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.31 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.55 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788146
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7779866, upper bound: 2.7788146
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7752518, upper bound: 2.7746755
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7782794, upper bound: 2.7746755
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7782361, upper bound: 2.7746755
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789702
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7789675
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7754041, upper bound: 2.7790491
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7781960, upper bound: 2.7790491
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788344
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7751457, upper bound: 2.7789023
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7755174
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7750874, upper bound: 2.7753317
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7770375, upper bound: 2.7746755
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7786160, upper bound: 2.7746755
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7747076, upper bound: 2.7746755
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7778045, upper bound: 2.7746755
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7783349
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753225
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782929
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7753309
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7783621
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7787676, upper bound: 2.7753216
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746913
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747043
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7747043, upper bound: 2.7746755
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746913, upper bound: 2.7746755
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7753216, upper bound: 2.7787676
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7783621, upper bound: 2.7787676
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7753309, upper bound: 2.7746755
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7782929, upper bound: 2.7746755
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7753225, upper bound: 2.7746755
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7783349, upper bound: 2.7746755
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7778045
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7747076
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7786160
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7770375
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7753317, upper bound: 2.7750874
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7755174, upper bound: 2.7746755
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7789023, upper bound: 2.7751457
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7788344, upper bound: 2.7746755
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7781960
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7790491, upper bound: 2.7754041
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7789675, upper bound: 2.7746755
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7789702, upper bound: 2.7746755
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782361
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7782794
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7752518
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7779866
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7788146, upper bound: 2.7746755
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.55
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533978
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7542692, upper bound: 2.7533778
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7560946, upper bound: 2.7533778
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533155
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533197
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532991, upper bound: 2.7533838
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535131, upper bound: 2.7563966
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535214, upper bound: 2.7564464
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575138
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575679
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535250, upper bound: 2.7562011
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535480, upper bound: 2.7563167
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577225
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7564821
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7555630
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7560268
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535687, upper bound: 2.7558570
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535718, upper bound: 2.7560691
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7565964, upper bound: 2.7532850
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7545022, upper bound: 2.7532625
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535629
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562925, upper bound: 2.7533260
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7548464, upper bound: 2.7533172
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7539970
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533279, upper bound: 2.7575015
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532883, upper bound: 2.7566529
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533996, upper bound: 2.7539491
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536538
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7536538, upper bound: 2.7532625
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541962, upper bound: 2.7532625
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541962, upper bound: 2.7532625
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7539491, upper bound: 2.7533996
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7566529, upper bound: 2.7532883
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575015, upper bound: 2.7533279
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577203, upper bound: 2.7532625
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7539970, upper bound: 2.7532625
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7532625
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7532625
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533172, upper bound: 2.7548464
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533260, upper bound: 2.7562925
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7545022
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532850, upper bound: 2.7565964
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535666, upper bound: 2.7535024
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535666, upper bound: 2.7535024
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534824, upper bound: 2.7533188
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7534824, upper bound: 2.7533188
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7532625
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7532625
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535729, upper bound: 2.7532625
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7535729, upper bound: 2.7532625
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7560691, upper bound: 2.7535718
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7558570, upper bound: 2.7535687
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7560268, upper bound: 2.7532625
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7555630, upper bound: 2.7532625
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7564821, upper bound: 2.7549012
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7564821, upper bound: 2.7549012
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7534652
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7534652
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7563167, upper bound: 2.7535480
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7562011, upper bound: 2.7535250
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575679, upper bound: 2.7532625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7575138, upper bound: 2.7532625
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7564464, upper bound: 2.7535214
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7563966, upper bound: 2.7535131
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7579100, upper bound: 2.7532625
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7579100, upper bound: 2.7532625
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7582459
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7582459
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7532991
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532625
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533197, upper bound: 2.7535127
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533155, upper bound: 2.7535127
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7560946
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7542692
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533978, upper bound: 2.7532625
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532625
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.31 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533978
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7542692, upper bound: 2.7533778
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7560946, upper bound: 2.7533778
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533155
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7535127, upper bound: 2.7533197
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7578181, upper bound: 2.7532967
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7533820
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532991, upper bound: 2.7533838
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7582459, upper bound: 2.7533354
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7579100
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7535131, upper bound: 2.7563966
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7535214, upper bound: 2.7564464
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575138
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7575679
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7535250, upper bound: 2.7562011
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7535480, upper bound: 2.7563167
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577225
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7534652, upper bound: 2.7577225
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7549012, upper bound: 2.7564821
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7564821
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7555630
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7560268
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7535687, upper bound: 2.7558570
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7535718, upper bound: 2.7560691
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535729
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536209
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533188, upper bound: 2.7534824
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7535024, upper bound: 2.7535666
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7565964, upper bound: 2.7532850
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7545022, upper bound: 2.7532625
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533225, upper bound: 2.7535629
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7535629
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7562925, upper bound: 2.7533260
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7548464, upper bound: 2.7533172
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7582293
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7539970
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7577203
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541317
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533279, upper bound: 2.7575015
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532883, upper bound: 2.7566529
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533996, upper bound: 2.7539491
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7541962
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7536538
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7536538, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7541962, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7541962, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7539491, upper bound: 2.7533996
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7566529, upper bound: 2.7532883
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7575015, upper bound: 2.7533279
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7541317, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7577203, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7539970, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7582293, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533172, upper bound: 2.7548464
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533260, upper bound: 2.7562925
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7535629, upper bound: 2.7533225
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7545022
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532850, upper bound: 2.7565964
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7535666, upper bound: 2.7535024
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7535666, upper bound: 2.7535024
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7534824, upper bound: 2.7533188
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7534824, upper bound: 2.7533188
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7536209, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7535729, upper bound: 2.7532625
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7535729, upper bound: 2.7532625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7560691, upper bound: 2.7535718
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7558570, upper bound: 2.7535687
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7560268, upper bound: 2.7532625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7555630, upper bound: 2.7532625
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7564821, upper bound: 2.7549012
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7564821, upper bound: 2.7549012
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7534652
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7577225, upper bound: 2.7534652
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7563167, upper bound: 2.7535480
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7562011, upper bound: 2.7535250
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7575679, upper bound: 2.7532625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7575138, upper bound: 2.7532625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7564464, upper bound: 2.7535214
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7563966, upper bound: 2.7535131
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7579100, upper bound: 2.7532625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7579100, upper bound: 2.7532625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7582459
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7582459
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533354, upper bound: 2.7532991
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532625
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532967, upper bound: 2.7578181
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533197, upper bound: 2.7535127
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533155, upper bound: 2.7535127
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7560946
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533778, upper bound: 2.7542692
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533978, upper bound: 2.7532625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7533820, upper bound: 2.7532625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.72
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.75 + 195.62 = 197.37 seconds
