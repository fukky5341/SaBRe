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
execution time: IAR + RelationalAnalysis = 0.80 + 0.97 = 1.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.7804846, upper bound: 2.7804846

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7804846, upper bound: 2.7803453
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7804846
time: 0.30 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.59 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -2.7804846, upper bound: 2.7803453
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.59
Output dim: 0, lower bound: -2.7803453, upper bound: 2.7804846

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7794709
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7794709, upper bound: 2.7794709
time: 0.28 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7801671
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7800387, upper bound: 2.7801671
time: 0.29 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.27 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.27
Output dim: 0, lower bound: -2.7797239, upper bound: 2.7794709
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.27
Output dim: 0, lower bound: -2.7794709, upper bound: 2.7794709
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.27
Output dim: 0, lower bound: -2.7771918, upper bound: 2.7801671
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.27
Output dim: 0, lower bound: -2.7800387, upper bound: 2.7801671

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7794964, upper bound: 2.7793131
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7794558, upper bound: 2.7756619
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7760038, upper bound: 2.7768017
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768629, upper bound: 2.7760307
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7800555
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7771264, upper bound: 2.7801435
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7793131, upper bound: 2.7794964
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7793131, upper bound: 2.7794964
time: 0.27 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.25 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -2.7794964, upper bound: 2.7793131
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -2.7794558, upper bound: 2.7756619
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -2.7760038, upper bound: 2.7768017
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -2.7768629, upper bound: 2.7760307
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7800555
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -2.7771264, upper bound: 2.7801435
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -2.7793131, upper bound: 2.7794964
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -2.7793131, upper bound: 2.7794964

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7794330, upper bound: 2.7763767
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7794964, upper bound: 2.7793131
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7607074, upper bound: 2.7581530
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7606543, upper bound: 2.7581530
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749982, upper bound: 2.7759671
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751479, upper bound: 2.7754546
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7766353, upper bound: 2.7759246
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7766353, upper bound: 2.7756619
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7800555
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7800240
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7538621, upper bound: 2.7585917
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7538569, upper bound: 2.7585917
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7793131, upper bound: 2.7794964
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756619, upper bound: 2.7792814
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7598270, upper bound: 2.7607410
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7598270, upper bound: 2.7607410
time: 0.33 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.33 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.33
Output dim: 0, lower bound: -2.7794330, upper bound: 2.7763767
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.33
Output dim: 0, lower bound: -2.7794964, upper bound: 2.7793131
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.33
Output dim: 0, lower bound: -2.7607074, upper bound: 2.7581530
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.33
Output dim: 0, lower bound: -2.7606543, upper bound: 2.7581530
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.33
Output dim: 0, lower bound: -2.7749982, upper bound: 2.7759671
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.33
Output dim: 0, lower bound: -2.7751479, upper bound: 2.7754546
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.33
Output dim: 0, lower bound: -2.7766353, upper bound: 2.7759246
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.33
Output dim: 0, lower bound: -2.7766353, upper bound: 2.7756619
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.33
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7800555
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.33
Output dim: 0, lower bound: -2.7768956, upper bound: 2.7800240
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.33
Output dim: 0, lower bound: -2.7538621, upper bound: 2.7585917
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.33
Output dim: 0, lower bound: -2.7538569, upper bound: 2.7585917
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.33
Output dim: 0, lower bound: -2.7793131, upper bound: 2.7794964
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.33
Output dim: 0, lower bound: -2.7756619, upper bound: 2.7792814
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.33
Output dim: 0, lower bound: -2.7598270, upper bound: 2.7607410
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.33
Output dim: 0, lower bound: -2.7598270, upper bound: 2.7607410

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748472, upper bound: 2.7747947
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7789927, upper bound: 2.7753030
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7792398
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7794346, upper bound: 2.7791934
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748889, upper bound: 2.7757444
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748889, upper bound: 2.7747947
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751479, upper bound: 2.7749654
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7751426, upper bound: 2.7754546
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7766353, upper bound: 2.7756619
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7766335, upper bound: 2.7759246
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7766353, upper bound: 2.7756619
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765113, upper bound: 2.7756619
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7540265, upper bound: 2.7582227
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7540265, upper bound: 2.7582227
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7540302, upper bound: 2.7578323
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7540302, upper bound: 2.7578870
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7759246, upper bound: 2.7766335
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765841, upper bound: 2.7758632
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7789274
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7748171
time: 0.28 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.31 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7748472, upper bound: 2.7747947
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7789927, upper bound: 2.7753030
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7792398
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7794346, upper bound: 2.7791934
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7748889, upper bound: 2.7757444
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7748889, upper bound: 2.7747947
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7751479, upper bound: 2.7749654
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7751426, upper bound: 2.7754546
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7766353, upper bound: 2.7756619
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7766335, upper bound: 2.7759246
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7766353, upper bound: 2.7756619
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7765113, upper bound: 2.7756619
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7540265, upper bound: 2.7582227
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7540265, upper bound: 2.7582227
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7540302, upper bound: 2.7578323
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7540302, upper bound: 2.7578870
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7759246, upper bound: 2.7766335
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7765841, upper bound: 2.7758632
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7789274
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.31
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7748171

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747043, upper bound: 2.7746755
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750124, upper bound: 2.7751352
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757974, upper bound: 2.7747947
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7764715
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7757209
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757080, upper bound: 2.7764157
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7765276, upper bound: 2.7757665
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480976
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480976
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748442, upper bound: 2.7753468
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750225, upper bound: 2.7753451
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7748171, upper bound: 2.7747947
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757974, upper bound: 2.7747947
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7755526, upper bound: 2.7747947
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7757974, upper bound: 2.7747947
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521619
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521619
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7749587, upper bound: 2.7757970
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7750801, upper bound: 2.7755600
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521671, upper bound: 2.7521619
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7521671, upper bound: 2.7521619
time: 0.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788344
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7748171
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7747983
time: 0.29 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.30 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7747043, upper bound: 2.7746755
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7750124, upper bound: 2.7751352
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7757974, upper bound: 2.7747947
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7764715
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7755129, upper bound: 2.7757209
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7757080, upper bound: 2.7764157
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7765276, upper bound: 2.7757665
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480976
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480976
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7748442, upper bound: 2.7753468
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7750225, upper bound: 2.7753451
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7748171, upper bound: 2.7747947
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7757974, upper bound: 2.7747947
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521671
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7755526, upper bound: 2.7747947
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7757974, upper bound: 2.7747947
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521619
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7521619, upper bound: 2.7521619
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7749587, upper bound: 2.7757970
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7750801, upper bound: 2.7755600
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7521671, upper bound: 2.7521619
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7521671, upper bound: 2.7521619
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7788344
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7748171
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.30
Output dim: 0, lower bound: -2.7747947, upper bound: 2.7747983

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746771, upper bound: 2.7746755
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746771, upper bound: 2.7746755
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756822, upper bound: 2.7746755
time: 0.33 seconds

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
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7748663
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746887
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7751603
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746913, upper bound: 2.7746755
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7756822, upper bound: 2.7746755
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754294, upper bound: 2.7746755
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7754335, upper bound: 2.7746755
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480976, upper bound: 2.7480598
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7480976, upper bound: 2.7480598
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7555630
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7560268
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746913
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746790
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
time: 0.29 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.42 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7746771, upper bound: 2.7746755
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7746771, upper bound: 2.7746755
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7756822, upper bound: 2.7746755
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7748663
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746887
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7516653, upper bound: 2.7516653
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7751603
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7746913, upper bound: 2.7746755
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7756822, upper bound: 2.7746755
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7754294, upper bound: 2.7746755
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7754335, upper bound: 2.7746755
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7480598, upper bound: 2.7480598
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7480976, upper bound: 2.7480598
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7480976, upper bound: 2.7480598
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7555630
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7560268
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7532625, upper bound: 2.7532625
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746913
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746790
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.42
Output dim: 0, lower bound: -2.7746755, upper bound: 2.7746755

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.5095432, 2.7755899, -0.5095432, 2.7755899, -3.2851331, 3.2851331
1: -0.5611423, 3.8165379, -0.5611423, 3.8165379, -4.3776803, 4.3776803
2: -1.3674926, 2.7016737, -1.3674926, 2.7016737, -4.0691662, 4.0691662
3: -1.1338987, 3.3577619, -1.1338987, 3.3577619, -4.4916606, 4.4916606
4: -1.7360522, 3.4053361, -1.7360522, 3.4053361, -5.1413884, 5.1413884

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
time: 0.31 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.41 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.41
Output dim: 0, lower bound: -2.7475631, upper bound: 2.7475631

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.77 + 107.63 = 109.40 seconds
