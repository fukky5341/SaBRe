## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 2.5152854478


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671)
1: (-1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297)
2: (-1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728)
3: (-1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408)
4: (-2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.23 + 1.29 = 2.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.5203261, upper bound: 2.5203261

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5201622, upper bound: 2.5201622
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5201622, upper bound: 2.5203102
time: 0.38 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.91 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.91
Output dim: 0, lower bound: -2.5201622, upper bound: 2.5201622
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.91
Output dim: 0, lower bound: -2.5201622, upper bound: 2.5203102

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5201374, upper bound: 2.5199410
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5199410, upper bound: 2.5198048
time: 0.44 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5198048, upper bound: 2.5200370
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5199410, upper bound: 2.5201374
time: 0.47 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.31 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -2.5201374, upper bound: 2.5199410
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -2.5199410, upper bound: 2.5198048
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -2.5198048, upper bound: 2.5200370
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.31
Output dim: 0, lower bound: -2.5199410, upper bound: 2.5201374

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170596, upper bound: 2.5170799
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170400, upper bound: 2.5171908
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170798, upper bound: 2.5169031
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170798, upper bound: 2.5170591
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170798
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170847
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170799, upper bound: 2.5170400
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170799, upper bound: 2.5170596
time: 0.35 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.99 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.99
Output dim: 0, lower bound: -2.5170596, upper bound: 2.5170799
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.99
Output dim: 0, lower bound: -2.5170400, upper bound: 2.5171908
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.99
Output dim: 0, lower bound: -2.5170798, upper bound: 2.5169031
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.99
Output dim: 0, lower bound: -2.5170798, upper bound: 2.5170591
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.99
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170798
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.99
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170847
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.99
Output dim: 0, lower bound: -2.5170799, upper bound: 2.5170400
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.99
Output dim: 0, lower bound: -2.5170799, upper bound: 2.5170596

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170799
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170400, upper bound: 2.5170802
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170386, upper bound: 2.5171908
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170798, upper bound: 2.5169031
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170798, upper bound: 2.5169031
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170746, upper bound: 2.5169031
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170386, upper bound: 2.5170591
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170798
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170746
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170798
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170847
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170802, upper bound: 2.5170386
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170802, upper bound: 2.5170400
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170799, upper bound: 2.5170596
time: 0.34 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170799
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -2.5170400, upper bound: 2.5170802
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -2.5170386, upper bound: 2.5171908
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -2.5170798, upper bound: 2.5169031
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -2.5170798, upper bound: 2.5169031
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -2.5170746, upper bound: 2.5169031
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -2.5170386, upper bound: 2.5170591
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170798
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170746
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170798
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170847
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -2.5170802, upper bound: 2.5170386
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -2.5170802, upper bound: 2.5170400
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.99
Output dim: 0, lower bound: -2.5170799, upper bound: 2.5170596

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170596, upper bound: 2.5170637
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170799
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170400, upper bound: 2.5170630
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170802
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170386, upper bound: 2.5170628
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170098, upper bound: 2.5171908
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170399, upper bound: 2.5169031
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170399, upper bound: 2.5169031
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170437, upper bound: 2.5169031
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170746, upper bound: 2.5169031
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169987, upper bound: 2.5169031
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170798, upper bound: 2.5169031
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170637, upper bound: 2.5170591
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170637
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170798
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169987
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170746
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170437
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170798
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170399
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170847
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170098
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170628, upper bound: 2.5170386
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170630, upper bound: 2.5169031
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170630, upper bound: 2.5170400
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170799, upper bound: 2.5169031
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170637, upper bound: 2.5170596
time: 0.38 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.06 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5170596, upper bound: 2.5170637
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170799
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5170400, upper bound: 2.5170630
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170802
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5170386, upper bound: 2.5170628
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5170098, upper bound: 2.5171908
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5170399, upper bound: 2.5169031
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5170399, upper bound: 2.5169031
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5170437, upper bound: 2.5169031
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5170746, upper bound: 2.5169031
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169987, upper bound: 2.5169031
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5170798, upper bound: 2.5169031
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5170637, upper bound: 2.5170591
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170637
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170798
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169987
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170746
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170437
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170798
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170399
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170847
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170098
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5170628, upper bound: 2.5170386
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5170630, upper bound: 2.5169031
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5170630, upper bound: 2.5170400
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5170799, upper bound: 2.5169031
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.06
Output dim: 0, lower bound: -2.5170637, upper bound: 2.5170596

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166317
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160254, upper bound: 2.5160178
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166491
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5161535
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5161789
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166320
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160254, upper bound: 2.5160178
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166496
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5167049
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160230, upper bound: 2.5161173
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5169426
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5164063
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160816
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166303
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166490
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5165633
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166467
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166116
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166493
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166078
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166870
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5163396, upper bound: 2.5165621
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169426, upper bound: 2.5160178
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166243
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160230
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5161857
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166275
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160254
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5161685
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166490, upper bound: 2.5160178
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166310
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166317, upper bound: 2.5160528
time: 0.40 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.68 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166317
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160254, upper bound: 2.5160178
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166491
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5161535
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5161789
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166320
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160254, upper bound: 2.5160178
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166496
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5167049
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160230, upper bound: 2.5161173
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5169426
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5164063
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160816
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166303
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166490
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5165633
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166467
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166116
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166493
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166078
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166870
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5163396, upper bound: 2.5165621
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5169426, upper bound: 2.5160178
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166243
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160230
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5161857
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166275
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160254
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5161685
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5166490, upper bound: 2.5160178
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166310
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.68
Output dim: 0, lower bound: -2.5166317, upper bound: 2.5160528

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142163
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142147
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148912
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148912
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139627
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139627
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142167
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142155
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142807
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142807
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142153
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142143
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148739
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148739
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139528
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139528
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142361
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142361
time: 0.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148592
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148592
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139468
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139468
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139468, upper bound: 2.5138278
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139468, upper bound: 2.5138278
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139468, upper bound: 2.5138278
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139468, upper bound: 2.5138278
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142205, upper bound: 2.5138278
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139528, upper bound: 2.5138278
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139528, upper bound: 2.5138278
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139528, upper bound: 2.5138278
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5148739, upper bound: 2.5138278
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143773, upper bound: 2.5138278
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143773, upper bound: 2.5138278
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142155, upper bound: 2.5138278
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142167, upper bound: 2.5138278
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139627, upper bound: 2.5138278
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139627, upper bound: 2.5138278
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5148912, upper bound: 2.5138278
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142163
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142147
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148912
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148912
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139627
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139627
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142167
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142155
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5143773
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5143773
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142153
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148739
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148739
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139528
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139528
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142205
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142202
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148592
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148592
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139468
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139468
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139468, upper bound: 2.5138278
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139468, upper bound: 2.5138278
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139468, upper bound: 2.5138278
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5148592, upper bound: 2.5138278
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139528, upper bound: 2.5138278
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139528, upper bound: 2.5138278
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142143, upper bound: 2.5138278
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5148739, upper bound: 2.5138278
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142155, upper bound: 2.5138278
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139627, upper bound: 2.5138278
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139627, upper bound: 2.5138278
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5148912, upper bound: 2.5138278
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5148912, upper bound: 2.5138278
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142147, upper bound: 2.5138278
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.43 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.42 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142163
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142147
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148912
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148912
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139627
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139627
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142167
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142155
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142807
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142807
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142153
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142143
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148739
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148739
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139528
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139528
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142361
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142361
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148592
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148592
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139468
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139468
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5139468, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5139468, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5139468, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5139468, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5142205, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5139528, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5139528, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5139528, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5148739, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5143773, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5143773, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5142155, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5142167, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5139627, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5139627, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5148912, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142163
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142147
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148912
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148912
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139627
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139627
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142167
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142155
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5143773
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5143773
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142153
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148739
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148739
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139528
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139528
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142205
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142202
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148592
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148592
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139468
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5139468
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5139468, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5139468, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5139468, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5148592, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5139528, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5139528, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5142143, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5148739, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5142155, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5139627, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5139627, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5148912, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5148912, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5142147, upper bound: 2.5138278
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.42
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.52 + 288.07 = 290.59 seconds
