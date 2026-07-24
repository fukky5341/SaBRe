## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 175.108430440685


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-43.8347244, 168.1222839, -43.8347244, 168.1222839, -211.9570007, 211.9570007)
1: (-119.0844269, 383.1814270, -119.0844269, 383.1814270, -502.2658691, 502.2658691)
2: (-175.4753113, 326.4086914, -175.4753113, 326.4086914, -501.8840027, 501.8840027)
3: (-100.4854202, 409.6264343, -100.4854202, 409.6264343, -510.1118469, 510.1118469)
4: (-160.2841492, 286.0808105, -160.2841492, 286.0808105, -446.3649597, 446.3649597)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.91 + 2.28 = 3.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -175.1171863, upper bound: 175.1171863

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
time: 0.78 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.65 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -43.8347244, 168.1222839, -43.8347244, 168.1222839, -211.9570007, 211.9570007
1: -119.0844269, 383.1814270, -119.0844269, 383.1814270, -502.2658691, 502.2658691
2: -175.4753113, 326.4086914, -175.4753113, 326.4086914, -501.8840027, 501.8840027
3: -100.4854202, 409.6264343, -100.4854202, 409.6264343, -510.1118469, 510.1118469
4: -160.2841492, 286.0808105, -160.2841492, 286.0808105, -446.3649597, 446.3649597

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
time: 0.78 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -43.8347244, 168.1222839, -43.8347244, 168.1222839, -211.9570007, 211.9570007
1: -119.0844269, 383.1814270, -119.0844269, 383.1814270, -502.2658691, 502.2658691
2: -175.4753113, 326.4086914, -175.4753113, 326.4086914, -501.8840027, 501.8840027
3: -100.4854202, 409.6264343, -100.4854202, 409.6264343, -510.1118469, 510.1118469
4: -160.2841492, 286.0808105, -160.2841492, 286.0808105, -446.3649597, 446.3649597

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
time: 0.78 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.79 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -175.1167515, upper bound: 175.1167515

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -43.8347244, 168.1222839, -43.8347244, 168.1222839, -211.9570007, 211.9570007
1: -119.0844269, 383.1814270, -119.0844269, 383.1814270, -502.2658691, 502.2658691
2: -175.4753113, 326.4086914, -175.4753113, 326.4086914, -501.8840027, 501.8840027
3: -100.4854202, 409.6264343, -100.4854202, 409.6264343, -510.1118469, 510.1118469
4: -160.2841492, 286.0808105, -160.2841492, 286.0808105, -446.3649597, 446.3649597

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1077670, upper bound: 175.1077663
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1077956, upper bound: 175.1077664
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -43.8347244, 168.1222839, -43.8347244, 168.1222839, -211.9570007, 211.9570007
1: -119.0844269, 383.1814270, -119.0844269, 383.1814270, -502.2658691, 502.2658691
2: -175.4753113, 326.4086914, -175.4753113, 326.4086914, -501.8840027, 501.8840027
3: -100.4854202, 409.6264343, -100.4854202, 409.6264343, -510.1118469, 510.1118469
4: -160.2841492, 286.0808105, -160.2841492, 286.0808105, -446.3649597, 446.3649597

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1077665, upper bound: 175.1077945
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1077670, upper bound: 175.1077945
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -43.8347244, 168.1222839, -43.8347244, 168.1222839, -211.9570007, 211.9570007
1: -119.0844269, 383.1814270, -119.0844269, 383.1814270, -502.2658691, 502.2658691
2: -175.4753113, 326.4086914, -175.4753113, 326.4086914, -501.8840027, 501.8840027
3: -100.4854202, 409.6264343, -100.4854202, 409.6264343, -510.1118469, 510.1118469
4: -160.2841492, 286.0808105, -160.2841492, 286.0808105, -446.3649597, 446.3649597

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1077665, upper bound: 175.1077670
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1077945, upper bound: 175.1077670
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -43.8347244, 168.1222839, -43.8347244, 168.1222839, -211.9570007, 211.9570007
1: -119.0844269, 383.1814270, -119.0844269, 383.1814270, -502.2658691, 502.2658691
2: -175.4753113, 326.4086914, -175.4753113, 326.4086914, -501.8840027, 501.8840027
3: -100.4854202, 409.6264343, -100.4854202, 409.6264343, -510.1118469, 510.1118469
4: -160.2841492, 286.0808105, -160.2841492, 286.0808105, -446.3649597, 446.3649597

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 19

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1077665, upper bound: 175.1077956
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -175.1077665, upper bound: 175.1077956
time: 0.96 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.83 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.83
Output dim: 0, lower bound: -175.1077670, upper bound: 175.1077663
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.83
Output dim: 0, lower bound: -175.1077956, upper bound: 175.1077664
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.83
Output dim: 0, lower bound: -175.1077665, upper bound: 175.1077945
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.83
Output dim: 0, lower bound: -175.1077670, upper bound: 175.1077945
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.83
Output dim: 0, lower bound: -175.1077665, upper bound: 175.1077670
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.83
Output dim: 0, lower bound: -175.1077945, upper bound: 175.1077670
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.83
Output dim: 0, lower bound: -175.1077665, upper bound: 175.1077956
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.83
Output dim: 0, lower bound: -175.1077665, upper bound: 175.1077956

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.19 + 18.62 = 21.82 seconds
