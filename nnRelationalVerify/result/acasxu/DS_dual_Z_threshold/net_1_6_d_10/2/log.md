## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.0038562720000000004


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551)
1: (-0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071)
2: (-0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934)
3: (-0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260)
4: (-0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.80 + 0.67 = 1.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0041916, upper bound: 0.0041916

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0041916, upper bound: 0.0041718
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0041718, upper bound: 0.0041916
time: 0.18 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.43 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.43
Output dim: 0, lower bound: -0.0041916, upper bound: 0.0041718
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.43
Output dim: 0, lower bound: -0.0041718, upper bound: 0.0041916

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040939, upper bound: 0.0040672
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040967, upper bound: 0.0040550
time: 0.17 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040550, upper bound: 0.0040967
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040672, upper bound: 0.0040939
time: 0.17 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.12 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.12
Output dim: 0, lower bound: -0.0040939, upper bound: 0.0040672
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.12
Output dim: 0, lower bound: -0.0040967, upper bound: 0.0040550
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.12
Output dim: 0, lower bound: -0.0040550, upper bound: 0.0040967
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.12
Output dim: 0, lower bound: -0.0040672, upper bound: 0.0040939

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040441, upper bound: 0.0039783
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040457, upper bound: 0.0039164
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039208, upper bound: 0.0039791
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040456, upper bound: 0.0039791
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039791, upper bound: 0.0040456
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039791, upper bound: 0.0039208
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039164, upper bound: 0.0040457
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039783, upper bound: 0.0040441
time: 0.16 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.12 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.12
Output dim: 0, lower bound: -0.0040441, upper bound: 0.0039783
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.12
Output dim: 0, lower bound: -0.0040457, upper bound: 0.0039164
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.12
Output dim: 0, lower bound: -0.0039208, upper bound: 0.0039791
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.12
Output dim: 0, lower bound: -0.0040456, upper bound: 0.0039791
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.12
Output dim: 0, lower bound: -0.0039791, upper bound: 0.0040456
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.12
Output dim: 0, lower bound: -0.0039791, upper bound: 0.0039208
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.12
Output dim: 0, lower bound: -0.0039164, upper bound: 0.0040457
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.12
Output dim: 0, lower bound: -0.0039783, upper bound: 0.0040441

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040017, upper bound: 0.0039428
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040178, upper bound: 0.0039120
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039386, upper bound: 0.0038795
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040180, upper bound: 0.0038698
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038583, upper bound: 0.0039449
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038820, upper bound: 0.0039291
time: 0.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039116, upper bound: 0.0039497
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040141, upper bound: 0.0039520
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039520, upper bound: 0.0040141
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039497, upper bound: 0.0039116
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039291, upper bound: 0.0038820
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039449, upper bound: 0.0038583
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038698, upper bound: 0.0040180
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038795, upper bound: 0.0039386
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039120, upper bound: 0.0040178
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039428, upper bound: 0.0040017
time: 0.17 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.13
Output dim: 0, lower bound: -0.0040017, upper bound: 0.0039428
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.13
Output dim: 0, lower bound: -0.0040178, upper bound: 0.0039120
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.13
Output dim: 0, lower bound: -0.0039386, upper bound: 0.0038795
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.13
Output dim: 0, lower bound: -0.0040180, upper bound: 0.0038698
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.13
Output dim: 0, lower bound: -0.0038583, upper bound: 0.0039449
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.13
Output dim: 0, lower bound: -0.0038820, upper bound: 0.0039291
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.13
Output dim: 0, lower bound: -0.0039116, upper bound: 0.0039497
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.13
Output dim: 0, lower bound: -0.0040141, upper bound: 0.0039520
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.13
Output dim: 0, lower bound: -0.0039520, upper bound: 0.0040141
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.13
Output dim: 0, lower bound: -0.0039497, upper bound: 0.0039116
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.13
Output dim: 0, lower bound: -0.0039291, upper bound: 0.0038820
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.13
Output dim: 0, lower bound: -0.0039449, upper bound: 0.0038583
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.13
Output dim: 0, lower bound: -0.0038698, upper bound: 0.0040180
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.13
Output dim: 0, lower bound: -0.0038795, upper bound: 0.0039386
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.13
Output dim: 0, lower bound: -0.0039120, upper bound: 0.0040178
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.13
Output dim: 0, lower bound: -0.0039428, upper bound: 0.0040017

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039811, upper bound: 0.0039428
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040017, upper bound: 0.0038920
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039463, upper bound: 0.0039120
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040178, upper bound: 0.0039014
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039386, upper bound: 0.0038795
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039375, upper bound: 0.0038583
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039431, upper bound: 0.0038698
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040180, upper bound: 0.0038583
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038583, upper bound: 0.0039449
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038583, upper bound: 0.0038825
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038583, upper bound: 0.0039291
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038820, upper bound: 0.0039108
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039084, upper bound: 0.0039497
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039116, upper bound: 0.0038707
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039236, upper bound: 0.0039520
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0040141, upper bound: 0.0039305
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039305, upper bound: 0.0040141
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039520, upper bound: 0.0039236
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038707, upper bound: 0.0039116
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039497, upper bound: 0.0039084
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039108, upper bound: 0.0038820
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039291, upper bound: 0.0038583
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038825, upper bound: 0.0038583
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039449, upper bound: 0.0038583
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038583, upper bound: 0.0040180
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038698, upper bound: 0.0039431
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038583, upper bound: 0.0039375
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038795, upper bound: 0.0039386
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039014, upper bound: 0.0040178
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039120, upper bound: 0.0039463
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038920, upper bound: 0.0040017
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0039428, upper bound: 0.0039811
time: 0.18 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039811, upper bound: 0.0039428
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0040017, upper bound: 0.0038920
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039463, upper bound: 0.0039120
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0040178, upper bound: 0.0039014
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039386, upper bound: 0.0038795
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039375, upper bound: 0.0038583
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039431, upper bound: 0.0038698
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0040180, upper bound: 0.0038583
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0038583, upper bound: 0.0039449
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0038583, upper bound: 0.0038825
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0038583, upper bound: 0.0039291
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0038820, upper bound: 0.0039108
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039084, upper bound: 0.0039497
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039116, upper bound: 0.0038707
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039236, upper bound: 0.0039520
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0040141, upper bound: 0.0039305
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039305, upper bound: 0.0040141
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039520, upper bound: 0.0039236
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0038707, upper bound: 0.0039116
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039497, upper bound: 0.0039084
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039108, upper bound: 0.0038820
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039291, upper bound: 0.0038583
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0038825, upper bound: 0.0038583
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039449, upper bound: 0.0038583
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0038583, upper bound: 0.0040180
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0038698, upper bound: 0.0039431
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0038583, upper bound: 0.0039375
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0038795, upper bound: 0.0039386
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039014, upper bound: 0.0040178
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039120, upper bound: 0.0039463
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0038920, upper bound: 0.0040017
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0039428, upper bound: 0.0039811

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038723, upper bound: 0.0037950
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038723, upper bound: 0.0037950
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038721, upper bound: 0.0037954
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038721, upper bound: 0.0037953
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038089, upper bound: 0.0037870
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038665, upper bound: 0.0037919
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038703, upper bound: 0.0038002
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038703, upper bound: 0.0037975
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038348, upper bound: 0.0037561
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038627, upper bound: 0.0037561
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038430, upper bound: 0.0037561
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038535, upper bound: 0.0037561
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038076, upper bound: 0.0037561
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038649, upper bound: 0.0037561
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038678, upper bound: 0.0037643
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038678, upper bound: 0.0037561
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0037667
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037645, upper bound: 0.0037753
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0037793
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0037793
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0037561
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037629, upper bound: 0.0037802
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038014
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037579, upper bound: 0.0037995
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038129, upper bound: 0.0037561
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038403, upper bound: 0.0037607
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038143, upper bound: 0.0037668
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038161, upper bound: 0.0037668
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038065, upper bound: 0.0037561
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038493, upper bound: 0.0037675
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038608, upper bound: 0.0038040
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038608, upper bound: 0.0038040
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038040, upper bound: 0.0038608
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038040, upper bound: 0.0038608
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037675, upper bound: 0.0038493
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038065
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037668, upper bound: 0.0038161
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037668, upper bound: 0.0038143
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037607, upper bound: 0.0038403
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038129
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037995, upper bound: 0.0037579
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038014, upper bound: 0.0037561
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037802, upper bound: 0.0037629
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0037561
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037793, upper bound: 0.0037561
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037793, upper bound: 0.0037561
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037753, upper bound: 0.0037645
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037667, upper bound: 0.0037561
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038678
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0037643, upper bound: 0.0038678
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038649
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038076
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038535
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038430
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038627
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038348
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0037975, upper bound: 0.0038703
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038002, upper bound: 0.0038703
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0037919, upper bound: 0.0038665
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037870, upper bound: 0.0038089
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0037953, upper bound: 0.0038721
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0037954, upper bound: 0.0038721
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0037950, upper bound: 0.0038723
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0037950, upper bound: 0.0038723
time: 0.18 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038723, upper bound: 0.0037950
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038723, upper bound: 0.0037950
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038721, upper bound: 0.0037954
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038721, upper bound: 0.0037953
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038089, upper bound: 0.0037870
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038665, upper bound: 0.0037919
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038703, upper bound: 0.0038002
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038703, upper bound: 0.0037975
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038348, upper bound: 0.0037561
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038627, upper bound: 0.0037561
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038430, upper bound: 0.0037561
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038535, upper bound: 0.0037561
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038076, upper bound: 0.0037561
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038649, upper bound: 0.0037561
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038678, upper bound: 0.0037643
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038678, upper bound: 0.0037561
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0037667
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037645, upper bound: 0.0037753
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0037793
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0037793
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0037561
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037629, upper bound: 0.0037802
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038014
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037579, upper bound: 0.0037995
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038129, upper bound: 0.0037561
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038403, upper bound: 0.0037607
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038143, upper bound: 0.0037668
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038161, upper bound: 0.0037668
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038065, upper bound: 0.0037561
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038493, upper bound: 0.0037675
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038608, upper bound: 0.0038040
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038608, upper bound: 0.0038040
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038040, upper bound: 0.0038608
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038040, upper bound: 0.0038608
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037675, upper bound: 0.0038493
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038065
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037668, upper bound: 0.0038161
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037668, upper bound: 0.0038143
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037607, upper bound: 0.0038403
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038129
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037995, upper bound: 0.0037579
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038014, upper bound: 0.0037561
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037802, upper bound: 0.0037629
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0037561
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037793, upper bound: 0.0037561
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037793, upper bound: 0.0037561
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037753, upper bound: 0.0037645
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037667, upper bound: 0.0037561
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038678
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037643, upper bound: 0.0038678
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038649
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038076
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038535
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038430
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038627
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037561, upper bound: 0.0038348
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037975, upper bound: 0.0038703
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0038002, upper bound: 0.0038703
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037919, upper bound: 0.0038665
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037870, upper bound: 0.0038089
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037953, upper bound: 0.0038721
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037954, upper bound: 0.0038721
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037950, upper bound: 0.0038723
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.72
Output dim: 0, lower bound: -0.0037950, upper bound: 0.0038723

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037876, upper bound: 0.0037688
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038390, upper bound: 0.0037856
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037872, upper bound: 0.0037688
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038407, upper bound: 0.0037856
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038032, upper bound: 0.0037891
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038455, upper bound: 0.0037891
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037980, upper bound: 0.0037781
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038455, upper bound: 0.0037869
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037680, upper bound: 0.0037653
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038371, upper bound: 0.0037833
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038448, upper bound: 0.0037903
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038513, upper bound: 0.0037903
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038348, upper bound: 0.0037903
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038513, upper bound: 0.0037903
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037995, upper bound: 0.0037281
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038321, upper bound: 0.0037410
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037899, upper bound: 0.0037281
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038360, upper bound: 0.0037410
time: 0.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038586, upper bound: 0.0037462
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038586, upper bound: 0.0037462
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038586, upper bound: 0.0037405
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038586, upper bound: 0.0037415
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038560, upper bound: 0.0037802
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038447, upper bound: 0.0037670
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038569, upper bound: 0.0037807
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0038569, upper bound: 0.0037735
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0037735, upper bound: 0.0038569
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0037807, upper bound: 0.0038569
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037670, upper bound: 0.0038447
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037802, upper bound: 0.0038560
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0037415, upper bound: 0.0038586
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0037405, upper bound: 0.0038586
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0037462, upper bound: 0.0038586
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0037462, upper bound: 0.0038586
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037410, upper bound: 0.0038360
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037281, upper bound: 0.0037899
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037410, upper bound: 0.0038321
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037281, upper bound: 0.0037995
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037903, upper bound: 0.0038513
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037903, upper bound: 0.0038348
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037903, upper bound: 0.0038513
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037903, upper bound: 0.0038448
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037833, upper bound: 0.0038371
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037653, upper bound: 0.0037680
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037869, upper bound: 0.0038455
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037781, upper bound: 0.0037980
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037891, upper bound: 0.0038455
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037891, upper bound: 0.0038032
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037856, upper bound: 0.0038407
time: 0.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037688, upper bound: 0.0037872
time: 0.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 15
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 14

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 3, pos: 15

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 14

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 47
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037856, upper bound: 0.0038390
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037688, upper bound: 0.0037876
time: 0.16 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.82 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037876, upper bound: 0.0037688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038390, upper bound: 0.0037856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037872, upper bound: 0.0037688
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038407, upper bound: 0.0037856
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038032, upper bound: 0.0037891
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038455, upper bound: 0.0037891
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037980, upper bound: 0.0037781
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038455, upper bound: 0.0037869
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037680, upper bound: 0.0037653
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038371, upper bound: 0.0037833
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038448, upper bound: 0.0037903
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038513, upper bound: 0.0037903
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038348, upper bound: 0.0037903
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038513, upper bound: 0.0037903
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037995, upper bound: 0.0037281
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038321, upper bound: 0.0037410
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037899, upper bound: 0.0037281
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038360, upper bound: 0.0037410
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038586, upper bound: 0.0037462
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038586, upper bound: 0.0037462
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038586, upper bound: 0.0037405
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038586, upper bound: 0.0037415
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038560, upper bound: 0.0037802
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038447, upper bound: 0.0037670
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038569, upper bound: 0.0037807
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0038569, upper bound: 0.0037735
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037735, upper bound: 0.0038569
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037807, upper bound: 0.0038569
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037670, upper bound: 0.0038447
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037802, upper bound: 0.0038560
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037415, upper bound: 0.0038586
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037405, upper bound: 0.0038586
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037462, upper bound: 0.0038586
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037462, upper bound: 0.0038586
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037410, upper bound: 0.0038360
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037281, upper bound: 0.0037899
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037410, upper bound: 0.0038321
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037281, upper bound: 0.0037995
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037903, upper bound: 0.0038513
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037903, upper bound: 0.0038348
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037903, upper bound: 0.0038513
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037903, upper bound: 0.0038448
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037833, upper bound: 0.0038371
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037653, upper bound: 0.0037680
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037869, upper bound: 0.0038455
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037781, upper bound: 0.0037980
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037891, upper bound: 0.0038455
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037891, upper bound: 0.0038032
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037856, upper bound: 0.0038407
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037688, upper bound: 0.0037872
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037856, upper bound: 0.0038390
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.82
Output dim: 0, lower bound: -0.0037688, upper bound: 0.0037876

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038247, upper bound: 0.0037090
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038152, upper bound: 0.0037104
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038247, upper bound: 0.0037090
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038152, upper bound: 0.0037104
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038247, upper bound: 0.0036933
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038152, upper bound: 0.0037051
time: 0.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038247, upper bound: 0.0037029
time: 0.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038152, upper bound: 0.0037058
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038232, upper bound: 0.0037309
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038127, upper bound: 0.0037470
time: 0.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038232, upper bound: 0.0037253
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0038127, upper bound: 0.0037414
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037414, upper bound: 0.0038127
time: 0.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037253, upper bound: 0.0038232
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037470, upper bound: 0.0038127
time: 0.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037309, upper bound: 0.0038232
time: 0.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037058, upper bound: 0.0038152
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037029, upper bound: 0.0038247
time: 0.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037051, upper bound: 0.0038152
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0036933, upper bound: 0.0038247
time: 0.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037104, upper bound: 0.0038152
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037090, upper bound: 0.0038247
time: 0.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0142358, -0.0094807, -0.0142358, -0.0094807, -0.0047551, 0.0047551
1: -0.0193883, -0.0168813, -0.0193883, -0.0168813, -0.0025071, 0.0025071
2: -0.0199593, -0.0157658, -0.0199593, -0.0157658, -0.0041934, 0.0041934
3: -0.0188365, -0.0065106, -0.0188365, -0.0065106, -0.0123260, 0.0123260
4: -0.0184974, -0.0067645, -0.0184974, -0.0067645, -0.0117329, 0.0117329

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 5
type: DSZ, layer: 5, pos: 28
type: DSZ, layer: 5, pos: 14
type: DSZ, layer: 5, pos: 24
type: DSZ, layer: 5, pos: 48
type: DSZ, layer: 5, pos: 49
type: DSZ, layer: 5, pos: 17
type: DSZ, layer: 5, pos: 3
type: DSZ, layer: 5, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 5, pos: 28

### Candidate
type: DSZ, layer: 5, pos: 14

### Candidate
type: DSZ, layer: 5, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 5, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037104, upper bound: 0.0038152
time: 0.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0037090, upper bound: 0.0038247
time: 0.18 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 1.55 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0038247, upper bound: 0.0037090
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0038152, upper bound: 0.0037104
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0038247, upper bound: 0.0037090
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0038152, upper bound: 0.0037104
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0038247, upper bound: 0.0036933
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0038152, upper bound: 0.0037051
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0038247, upper bound: 0.0037029
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0038152, upper bound: 0.0037058
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0038232, upper bound: 0.0037309
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0038127, upper bound: 0.0037470
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0038232, upper bound: 0.0037253
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0038127, upper bound: 0.0037414
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0037414, upper bound: 0.0038127
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0037253, upper bound: 0.0038232
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0037470, upper bound: 0.0038127
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0037309, upper bound: 0.0038232
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0037058, upper bound: 0.0038152
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0037029, upper bound: 0.0038247
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0037051, upper bound: 0.0038152
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0036933, upper bound: 0.0038247
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0037104, upper bound: 0.0038152
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0037090, upper bound: 0.0038247
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0037104, upper bound: 0.0038152
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 1.55
Output dim: 0, lower bound: -0.0037090, upper bound: 0.0038247

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.47 + 152.78 = 154.25 seconds
