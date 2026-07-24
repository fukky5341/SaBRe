## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 187.542370087


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746)
1: (-117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561)
2: (-169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212)
3: (-63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962)
4: (-188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.85 + 1.61 = 2.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -187.9182065, upper bound: 187.9182065

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.9167375, upper bound: 187.9167375
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.9167375, upper bound: 187.9173613
time: 0.53 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.20 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.20
Output dim: 3, lower bound: -187.9167375, upper bound: 187.9167375
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.20
Output dim: 3, lower bound: -187.9167375, upper bound: 187.9173613

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6715312, upper bound: 187.6764939
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6766269, upper bound: 187.6717917
time: 0.55 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.9109065, upper bound: 187.8968952
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8992479, upper bound: 187.9110769
time: 0.53 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.87 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.87
Output dim: 3, lower bound: -187.6715312, upper bound: 187.6764939
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.87
Output dim: 3, lower bound: -187.6766269, upper bound: 187.6717917
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.87
Output dim: 3, lower bound: -187.9109065, upper bound: 187.8968952
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.87
Output dim: 3, lower bound: -187.8992479, upper bound: 187.9110769

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6690026, upper bound: 187.6745092
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6709621, upper bound: 187.6725839
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6725264, upper bound: 187.6705234
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6761378, upper bound: 187.6694194
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.9102968, upper bound: 187.8968952
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.9109065, upper bound: 187.8936206
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8961492, upper bound: 187.9084147
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8910328, upper bound: 187.9084105
time: 0.55 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.79 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.79
Output dim: 3, lower bound: -187.6690026, upper bound: 187.6745092
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.79
Output dim: 3, lower bound: -187.6709621, upper bound: 187.6725839
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.79
Output dim: 3, lower bound: -187.6725264, upper bound: 187.6705234
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.79
Output dim: 3, lower bound: -187.6761378, upper bound: 187.6694194
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.79
Output dim: 3, lower bound: -187.9102968, upper bound: 187.8968952
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.79
Output dim: 3, lower bound: -187.9109065, upper bound: 187.8936206
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.79
Output dim: 3, lower bound: -187.8961492, upper bound: 187.9084147
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.79
Output dim: 3, lower bound: -187.8910328, upper bound: 187.9084105

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6661153, upper bound: 187.6585982
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6576238, upper bound: 187.6693331
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6650509, upper bound: 187.6725839
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6709621, upper bound: 187.6642964
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6725083, upper bound: 187.6705234
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6725264, upper bound: 187.6581634
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6699120, upper bound: 187.6584442
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6583962, upper bound: 187.6661983
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.9036067, upper bound: 187.8888197
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.9043310, upper bound: 187.8908263
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8898067, upper bound: 187.8936206
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.9109065, upper bound: 187.8843994
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8369786, upper bound: 187.8403367
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8369786, upper bound: 187.8403367
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8369786, upper bound: 187.8409569
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8377821, upper bound: 187.8409569
time: 0.53 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.91 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 3, lower bound: -187.6661153, upper bound: 187.6585982
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 3, lower bound: -187.6576238, upper bound: 187.6693331
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 3, lower bound: -187.6650509, upper bound: 187.6725839
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 3, lower bound: -187.6709621, upper bound: 187.6642964
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 3, lower bound: -187.6725083, upper bound: 187.6705234
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 3, lower bound: -187.6725264, upper bound: 187.6581634
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 3, lower bound: -187.6699120, upper bound: 187.6584442
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 3, lower bound: -187.6583962, upper bound: 187.6661983
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 3, lower bound: -187.9036067, upper bound: 187.8888197
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 3, lower bound: -187.9043310, upper bound: 187.8908263
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 3, lower bound: -187.8898067, upper bound: 187.8936206
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 3, lower bound: -187.9109065, upper bound: 187.8843994
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 3, lower bound: -187.8369786, upper bound: 187.8403367
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 3, lower bound: -187.8369786, upper bound: 187.8403367
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 3, lower bound: -187.8369786, upper bound: 187.8409569
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.91
Output dim: 3, lower bound: -187.8377821, upper bound: 187.8409569

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6568148, upper bound: 187.6579703
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6661153, upper bound: 187.6581446
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6574538, upper bound: 187.6693331
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6574721, upper bound: 187.6595720
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6635485, upper bound: 187.6725839
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6650316, upper bound: 187.6718650
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300978, upper bound: 187.6258314
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300770, upper bound: 187.6167778
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6581634, upper bound: 187.6705234
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6725083, upper bound: 187.6650997
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6677193, upper bound: 187.6569955
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6577461, upper bound: 187.6569955
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6697792, upper bound: 187.6584442
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6698571, upper bound: 187.6574390
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6170162, upper bound: 187.6267011
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6189188, upper bound: 187.6260637
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8340334, upper bound: 187.8349074
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8353673, upper bound: 187.8349074
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8794932, upper bound: 187.8908263
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.9043310, upper bound: 187.8806450
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8733802, upper bound: 187.8741556
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8851708, upper bound: 187.8742723
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8222177, upper bound: 187.8222177
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8222177, upper bound: 187.8222177
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7875165, upper bound: 187.7875165
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7875165, upper bound: 187.7875165
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8258452, upper bound: 187.8392109
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8258452, upper bound: 187.8259503
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8370922, upper bound: 187.8402660
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8363178, upper bound: 187.8399055
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8375649, upper bound: 187.8409569
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8377821, upper bound: 187.8369957
time: 0.51 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.97 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.6568148, upper bound: 187.6579703
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.6661153, upper bound: 187.6581446
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.6574538, upper bound: 187.6693331
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.6574721, upper bound: 187.6595720
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.6635485, upper bound: 187.6725839
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.6650316, upper bound: 187.6718650
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.6300978, upper bound: 187.6258314
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.6300770, upper bound: 187.6167778
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.6581634, upper bound: 187.6705234
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.6725083, upper bound: 187.6650997
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.6677193, upper bound: 187.6569955
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.6577461, upper bound: 187.6569955
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.6697792, upper bound: 187.6584442
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.6698571, upper bound: 187.6574390
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.6170162, upper bound: 187.6267011
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.6189188, upper bound: 187.6260637
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.8340334, upper bound: 187.8349074
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.8353673, upper bound: 187.8349074
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.8794932, upper bound: 187.8908263
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.9043310, upper bound: 187.8806450
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.8733802, upper bound: 187.8741556
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.8851708, upper bound: 187.8742723
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.8222177, upper bound: 187.8222177
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.8222177, upper bound: 187.8222177
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.7875165, upper bound: 187.7875165
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.7875165, upper bound: 187.7875165
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.8258452, upper bound: 187.8392109
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.8258452, upper bound: 187.8259503
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.8370922, upper bound: 187.8402660
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.8363178, upper bound: 187.8399055
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.8375649, upper bound: 187.8409569
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.97
Output dim: 3, lower bound: -187.8377821, upper bound: 187.8369957

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6578620
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6575588
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6660710, upper bound: 187.6580841
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6661153, upper bound: 187.6574579
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168190, upper bound: 187.6314444
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6263951
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6595720
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6574721, upper bound: 187.6569955
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6725367
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6635485, upper bound: 187.6725839
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6718650
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6650316, upper bound: 187.6718156
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6293701, upper bound: 187.6258314
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300787, upper bound: 187.6189947
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6199998, upper bound: 187.6167778
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300757, upper bound: 187.6167778
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6580863
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6669851
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279032, upper bound: 187.6265049
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6323326, upper bound: 187.6256475
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6569955
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6677193, upper bound: 187.6569955
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6576567, upper bound: 187.6569955
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6577461, upper bound: 187.6569955
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6643869, upper bound: 187.6577646
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6697792, upper bound: 187.6579483
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6670305, upper bound: 187.6569955
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6698571, upper bound: 187.6574390
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6181121, upper bound: 187.6267011
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6170162, upper bound: 187.6223829
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6186213, upper bound: 187.6260637
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6175698, upper bound: 187.6213709
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8228427, upper bound: 187.8249830
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8281545, upper bound: 187.8224870
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8307670, upper bound: 187.8308877
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8322104, upper bound: 187.8315277
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6469286, upper bound: 187.6443558
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6469286, upper bound: 187.6437336
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8331601, upper bound: 187.8341434
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8331601, upper bound: 187.8341434
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8169410, upper bound: 187.8172764
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8169410, upper bound: 187.8172764
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6641072, upper bound: 187.6578145
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6641072, upper bound: 187.6576238
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8169410, upper bound: 187.8169410
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8169410, upper bound: 187.8169410
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8119741, upper bound: 187.8119741
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8119741, upper bound: 187.8119741
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7875165, upper bound: 187.7875165
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7875165, upper bound: 187.7875165
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8256970
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8385110
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8258452, upper bound: 187.8259503
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8258452, upper bound: 187.8258866
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8368729, upper bound: 187.8402660
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8370922, upper bound: 187.8363178
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8390848
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8253994
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7875165, upper bound: 187.7875445
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7875165, upper bound: 187.7875177
time: 0.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8370922, upper bound: 187.8363178
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8363178, upper bound: 187.8363308
time: 0.68 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.38 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6578620
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6575588
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6660710, upper bound: 187.6580841
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6661153, upper bound: 187.6574579
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6168190, upper bound: 187.6314444
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6263951
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6595720
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6574721, upper bound: 187.6569955
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6725367
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6635485, upper bound: 187.6725839
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6718650
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6650316, upper bound: 187.6718156
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6293701, upper bound: 187.6258314
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6300787, upper bound: 187.6189947
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6199998, upper bound: 187.6167778
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6300757, upper bound: 187.6167778
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6580863
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6669851
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6279032, upper bound: 187.6265049
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6323326, upper bound: 187.6256475
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6569955
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6677193, upper bound: 187.6569955
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6576567, upper bound: 187.6569955
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6577461, upper bound: 187.6569955
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6643869, upper bound: 187.6577646
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6697792, upper bound: 187.6579483
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6670305, upper bound: 187.6569955
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6698571, upper bound: 187.6574390
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6181121, upper bound: 187.6267011
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6170162, upper bound: 187.6223829
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6186213, upper bound: 187.6260637
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6175698, upper bound: 187.6213709
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8228427, upper bound: 187.8249830
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8281545, upper bound: 187.8224870
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8307670, upper bound: 187.8308877
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8322104, upper bound: 187.8315277
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6469286, upper bound: 187.6443558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6469286, upper bound: 187.6437336
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8331601, upper bound: 187.8341434
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8331601, upper bound: 187.8341434
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8169410, upper bound: 187.8172764
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8169410, upper bound: 187.8172764
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6641072, upper bound: 187.6578145
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.6641072, upper bound: 187.6576238
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8169410, upper bound: 187.8169410
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8169410, upper bound: 187.8169410
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8119741, upper bound: 187.8119741
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8119741, upper bound: 187.8119741
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.7875165, upper bound: 187.7875165
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.7875165, upper bound: 187.7875165
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8256970
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8385110
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8258452, upper bound: 187.8259503
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8258452, upper bound: 187.8258866
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8368729, upper bound: 187.8402660
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8370922, upper bound: 187.8363178
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8390848
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8253994
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.7875165, upper bound: 187.7875445
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.7875165, upper bound: 187.7875177
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8370922, upper bound: 187.8363178
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 3, lower bound: -187.8363178, upper bound: 187.8363308

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6173365
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6170929
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6659157, upper bound: 187.6579615
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6660710, upper bound: 187.6580841
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259685, upper bound: 187.6169299
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6314444
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168190, upper bound: 187.6256842
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6263951
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6263307
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6203960
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168405, upper bound: 187.6163102
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168405, upper bound: 187.6163102
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6576058
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6677462
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6635446, upper bound: 187.6725839
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6578392
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6575576
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6674108
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6267311, upper bound: 187.6316265
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266741, upper bound: 187.6274766
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6194624
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6293701, upper bound: 187.6258314
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277696, upper bound: 187.6169003
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6171977, upper bound: 187.6183588
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6167778, upper bound: 187.6167778
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6199998, upper bound: 187.6167778
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6276451, upper bound: 187.6167778
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6300757, upper bound: 187.6167778
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6578332
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6580863
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6669851
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6668895
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6264715
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279032, upper bound: 187.6261777
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6316206, upper bound: 187.6256475
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6323326, upper bound: 187.6250123
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6569955
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6569955
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259963, upper bound: 187.6163102
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6286009, upper bound: 187.6163102
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6569955
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6576567, upper bound: 187.6569955
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168195, upper bound: 187.6163102
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6566690
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6643869, upper bound: 187.6577646
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6645174, upper bound: 187.6579483
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6697792, upper bound: 187.6579429
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6569955
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6670305, upper bound: 187.6569955
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6278244, upper bound: 187.6166131
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6324002, upper bound: 187.6167864
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165013, upper bound: 187.6165013
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165013, upper bound: 187.6266759
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6170321, upper bound: 187.6165013
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165013, upper bound: 187.6223494
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6161418, upper bound: 187.6260637
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6161418, upper bound: 187.6179328
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6173896, upper bound: 187.6213709
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6173597, upper bound: 187.6159535
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8219772, upper bound: 187.8242874
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8219795, upper bound: 187.8240632
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8272484, upper bound: 187.8217975
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8217975, upper bound: 187.8217975
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8299479, upper bound: 187.8297889
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8299689, upper bound: 187.8302113
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8313725, upper bound: 187.8308513
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8300414, upper bound: 187.8306363
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6436665, upper bound: 187.6442668
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6469265, upper bound: 187.6440681
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6384472, upper bound: 187.6384472
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6434728, upper bound: 187.6384472
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8217975, upper bound: 187.8242691
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8217975, upper bound: 187.8217975
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8217975, upper bound: 187.8242691
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8217975, upper bound: 187.8217975
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6637238, upper bound: 187.6575583
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6641072, upper bound: 187.6570513
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6639611, upper bound: 187.6573075
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6641072, upper bound: 187.6568148
time: 0.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8120837, upper bound: 187.8120837
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8120837, upper bound: 187.8120837
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8066874, upper bound: 187.8066874
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8066874, upper bound: 187.8066874
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8071023, upper bound: 187.8071023
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8071023, upper bound: 187.8071023
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8066874, upper bound: 187.8066874
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8066874, upper bound: 187.8066874
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8256970
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8251844
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8385110
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8252688
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7799738, upper bound: 187.7799738
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7799738, upper bound: 187.7799738
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7799738, upper bound: 187.7799738
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7799738, upper bound: 187.7799738
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7859102
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858871
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8251844
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8276448, upper bound: 187.8251844
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7828197
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7828197
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7859102
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7787825, upper bound: 187.7787825
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.7787825, upper bound: 187.7787855
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8251844
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8251844
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8297889, upper bound: 187.8298054
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.8297889, upper bound: 187.8297957
time: 0.55 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.46 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6173365
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6170929
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6659157, upper bound: 187.6579615
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6660710, upper bound: 187.6580841
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6259685, upper bound: 187.6169299
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6314444
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6168190, upper bound: 187.6256842
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6263951
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6263307
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6203960
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6168405, upper bound: 187.6163102
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6168405, upper bound: 187.6163102
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6576058
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6677462
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6635446, upper bound: 187.6725839
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6578392, upper bound: 187.6578392
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6575576
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6674108
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6267311, upper bound: 187.6316265
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6266741, upper bound: 187.6274766
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6194624
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6293701, upper bound: 187.6258314
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6277696, upper bound: 187.6169003
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6171977, upper bound: 187.6183588
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6167778, upper bound: 187.6167778
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6199998, upper bound: 187.6167778
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6276451, upper bound: 187.6167778
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6300757, upper bound: 187.6167778
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6578332
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6580863
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6669851
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6668895
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6264715
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6279032, upper bound: 187.6261777
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6316206, upper bound: 187.6256475
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6323326, upper bound: 187.6250123
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6569955
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6569955
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6259963, upper bound: 187.6163102
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6286009, upper bound: 187.6163102
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6569955
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6576567, upper bound: 187.6569955
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6168195, upper bound: 187.6163102
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6566690
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6643869, upper bound: 187.6577646
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6645174, upper bound: 187.6579483
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6697792, upper bound: 187.6579429
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6569955, upper bound: 187.6569955
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6670305, upper bound: 187.6569955
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6278244, upper bound: 187.6166131
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6324002, upper bound: 187.6167864
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6165013, upper bound: 187.6165013
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6165013, upper bound: 187.6266759
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6170321, upper bound: 187.6165013
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6165013, upper bound: 187.6223494
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6161418, upper bound: 187.6260637
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6161418, upper bound: 187.6179328
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6173896, upper bound: 187.6213709
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6173597, upper bound: 187.6159535
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8219772, upper bound: 187.8242874
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8219795, upper bound: 187.8240632
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8272484, upper bound: 187.8217975
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8217975, upper bound: 187.8217975
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8299479, upper bound: 187.8297889
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8299689, upper bound: 187.8302113
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8313725, upper bound: 187.8308513
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8300414, upper bound: 187.8306363
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6436665, upper bound: 187.6442668
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6469265, upper bound: 187.6440681
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6384472, upper bound: 187.6384472
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6434728, upper bound: 187.6384472
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8217975, upper bound: 187.8242691
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8217975, upper bound: 187.8217975
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8217975, upper bound: 187.8242691
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8217975, upper bound: 187.8217975
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7829591, upper bound: 187.7829591
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6637238, upper bound: 187.6575583
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6641072, upper bound: 187.6570513
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6639611, upper bound: 187.6573075
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.6641072, upper bound: 187.6568148
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8120837, upper bound: 187.8120837
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8120837, upper bound: 187.8120837
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8066874, upper bound: 187.8066874
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8066874, upper bound: 187.8066874
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8071023, upper bound: 187.8071023
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8071023, upper bound: 187.8071023
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8066874, upper bound: 187.8066874
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8066874, upper bound: 187.8066874
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8256970
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8251844
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8385110
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8252688
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7799738, upper bound: 187.7799738
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7799738, upper bound: 187.7799738
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7799738, upper bound: 187.7799738
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7799738, upper bound: 187.7799738
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7859102
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858871
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8251844
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8276448, upper bound: 187.8251844
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7828197
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7828197
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7783802, upper bound: 187.7783802
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7859102
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7858861, upper bound: 187.7858861
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7787825, upper bound: 187.7787825
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.7787825, upper bound: 187.7787855
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8251844
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8251844, upper bound: 187.8251844
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8297889, upper bound: 187.8298054
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.46
Output dim: 3, lower bound: -187.8297889, upper bound: 187.8297957

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6173365
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6173216
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6170929
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6579615
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6659157, upper bound: 187.6578713
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6258039, upper bound: 187.6186697
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6169299
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259685, upper bound: 187.6159535
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165724, upper bound: 187.6159535
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6314444
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6278755
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168115, upper bound: 187.6256842
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168190, upper bound: 187.6251949
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6263951
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6263307
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6203960
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168405, upper bound: 187.6163102
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6171512
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6286284
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6324825
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6282844
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6566690
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6566690
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6575576
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6566690
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6674108
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6566690
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256917, upper bound: 187.6169998
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6178938, upper bound: 187.6281658
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6256428, upper bound: 187.6164504
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6182895, upper bound: 187.6256926
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6170328
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6187848
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6258314
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6293701, upper bound: 187.6230959
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6261724, upper bound: 187.6169003
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277696, upper bound: 187.6167652
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6171977, upper bound: 187.6183588
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6169853, upper bound: 187.6159535
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6161418, upper bound: 187.6161418
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6161418, upper bound: 187.6161418
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6193075, upper bound: 187.6161418
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6167549, upper bound: 187.6161418
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6266110, upper bound: 187.6161418
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6173590, upper bound: 187.6161418
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277626, upper bound: 187.6161418
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6171895, upper bound: 187.6161418
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6174118
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6172923
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6174837
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6172710
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6669851
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6576538
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6274533
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6265395
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6264715
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6165894, upper bound: 187.6260656
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277006, upper bound: 187.6261777
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6279032, upper bound: 187.6259919
time: 0.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6281618, upper bound: 187.6183520
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6170433, upper bound: 187.6246623
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6189178, upper bound: 187.6249867
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6323326, upper bound: 187.6247816
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259949, upper bound: 187.6163102
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6259758, upper bound: 187.6163102
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6159535, upper bound: 187.6159535
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6286009, upper bound: 187.6159535
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6566690
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6566690
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168268, upper bound: 187.6163102
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6173716, upper bound: 187.6163102
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6566690
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6566690
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6630988, upper bound: 187.6577646
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6643869, upper bound: 187.6567719
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6265526, upper bound: 187.6171593
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6269392, upper bound: 187.6167545
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6277681, upper bound: 187.6174459
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6321630, upper bound: 187.6167875
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6566690, upper bound: 187.6566690
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6670305, upper bound: 187.6566690
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6268313, upper bound: 187.6166131
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6278244, upper bound: 187.6163102
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6290599, upper bound: 187.6164501
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6324002, upper bound: 187.6159535
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6161418, upper bound: 187.6161418
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6161418, upper bound: 187.6161418
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6178248, upper bound: 187.6266759
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6179519, upper bound: 187.6185118
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6163102, upper bound: 187.6163102
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6168554, upper bound: 187.6163102
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6161418, upper bound: 187.6161418
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -187.6171489, upper bound: 187.6222341
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.6440735, 126.7424088, -149.6440735, 126.7424088, -276.3864746, 276.3864746
1: -117.3338928, 118.4335785, -117.3338928, 118.4335785, -235.7674713, 235.7674561
2: -169.7016296, 131.6250763, -169.7016296, 131.6250763, -301.3267212, 301.3267212
3: -63.3496017, 169.2627869, -63.3496017, 169.2627869, -232.6123962, 232.6123962
4: -188.6523895, 133.4867859, -188.6523895, 133.4867859, -322.1391602, 322.1391602

Time for backsubstitution: 1.13 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.46 + 417.98 = 420.43 seconds
