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
execution time: IAR + RelationalAnalysis = 1.03 + 1.11 = 2.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.5203261, upper bound: 2.5203261

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5190530, upper bound: 2.5189611
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5189611, upper bound: 2.5190530
time: 0.40 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.76 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -2.5190530, upper bound: 2.5189611
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -2.5189611, upper bound: 2.5190530

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5189598, upper bound: 2.5189611
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5189611, upper bound: 2.5189598
time: 0.35 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5189611, upper bound: 2.5190517
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5189611, upper bound: 2.5190530
time: 0.32 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.59 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.59
Output dim: 0, lower bound: -2.5189598, upper bound: 2.5189611
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.59
Output dim: 0, lower bound: -2.5189611, upper bound: 2.5189598
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.59
Output dim: 0, lower bound: -2.5189611, upper bound: 2.5190517
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.59
Output dim: 0, lower bound: -2.5189611, upper bound: 2.5190530

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5189598, upper bound: 2.5189611
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5189150, upper bound: 2.5189611
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176255, upper bound: 2.5174606
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176256, upper bound: 2.5176249
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5184975, upper bound: 2.5185947
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5185014, upper bound: 2.5184481
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5189150, upper bound: 2.5189615
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5189611, upper bound: 2.5190530
time: 0.38 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.64 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 0, lower bound: -2.5189598, upper bound: 2.5189611
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 0, lower bound: -2.5189150, upper bound: 2.5189611
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 0, lower bound: -2.5176255, upper bound: 2.5174606
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 0, lower bound: -2.5176256, upper bound: 2.5176249
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 0, lower bound: -2.5184975, upper bound: 2.5185947
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 0, lower bound: -2.5185014, upper bound: 2.5184481
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 0, lower bound: -2.5189150, upper bound: 2.5189615
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.64
Output dim: 0, lower bound: -2.5189611, upper bound: 2.5190530

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176255, upper bound: 2.5176177
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5177285, upper bound: 2.5176167
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5175985, upper bound: 2.5176253
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5175985, upper bound: 2.5176253
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176010, upper bound: 2.5174548
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176010, upper bound: 2.5174606
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170746
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170802, upper bound: 2.5170400
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5184481, upper bound: 2.5185018
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5184481, upper bound: 2.5185947
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170798, upper bound: 2.5169031
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170799, upper bound: 2.5169031
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5184481, upper bound: 2.5185017
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5184650, upper bound: 2.5184481
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176167, upper bound: 2.5177285
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5176177, upper bound: 2.5176634
time: 0.41 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.75 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -2.5176255, upper bound: 2.5176177
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -2.5177285, upper bound: 2.5176167
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -2.5175985, upper bound: 2.5176253
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -2.5175985, upper bound: 2.5176253
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -2.5176010, upper bound: 2.5174548
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -2.5176010, upper bound: 2.5174606
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170746
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -2.5170802, upper bound: 2.5170400
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -2.5184481, upper bound: 2.5185018
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -2.5184481, upper bound: 2.5185947
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -2.5170798, upper bound: 2.5169031
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -2.5170799, upper bound: 2.5169031
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -2.5184481, upper bound: 2.5185017
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -2.5184650, upper bound: 2.5184481
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -2.5176167, upper bound: 2.5177285
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.75
Output dim: 0, lower bound: -2.5176177, upper bound: 2.5176634

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170596, upper bound: 2.5170637
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170746, upper bound: 2.5169031
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170591, upper bound: 2.5170637
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5171908, upper bound: 2.5170098
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170799
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170399, upper bound: 2.5169031
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170798
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170628, upper bound: 2.5170386
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170798, upper bound: 2.5169031
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170437, upper bound: 2.5169031
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169987
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170746
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170630, upper bound: 2.5169031
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170630, upper bound: 2.5170400
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170400, upper bound: 2.5170630
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170437
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170386, upper bound: 2.5170628
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170399
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170798, upper bound: 2.5169031
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170799, upper bound: 2.5169031
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170802
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170798
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169987, upper bound: 2.5169031
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170098, upper bound: 2.5171908
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170637, upper bound: 2.5170591
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170847
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5170637, upper bound: 2.5170596
time: 0.37 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.73 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5170596, upper bound: 2.5170637
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5170746, upper bound: 2.5169031
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5170591, upper bound: 2.5170637
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5171908, upper bound: 2.5170098
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170799
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5170399, upper bound: 2.5169031
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170798
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5170628, upper bound: 2.5170386
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5170798, upper bound: 2.5169031
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5170437, upper bound: 2.5169031
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169987
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170746
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5170630, upper bound: 2.5169031
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5170630, upper bound: 2.5170400
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5170400, upper bound: 2.5170630
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170437
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5170386, upper bound: 2.5170628
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170399
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5170798, upper bound: 2.5169031
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5170799, upper bound: 2.5169031
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170802
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170798
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5169987, upper bound: 2.5169031
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5169031
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5170098, upper bound: 2.5171908
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5170637, upper bound: 2.5170591
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5169031, upper bound: 2.5170847
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.73
Output dim: 0, lower bound: -2.5170637, upper bound: 2.5170596

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166317
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160254, upper bound: 2.5160178
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5161097, upper bound: 2.5156873
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160979, upper bound: 2.5156873
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5168390, upper bound: 2.5166868
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5168428
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145783, upper bound: 2.5144801
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5150510, upper bound: 2.5144801
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5144801, upper bound: 2.5150809
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5144801, upper bound: 2.5150809
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5166824
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5168596
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160522, upper bound: 2.5156873
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160504, upper bound: 2.5156873
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5161535
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166493, upper bound: 2.5160178
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5161789
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5144801, upper bound: 2.5144801
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5144801, upper bound: 2.5144801
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5153302, upper bound: 2.5156261
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5153302, upper bound: 2.5153302
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5156873, upper bound: 2.5160138
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5156873, upper bound: 2.5157136
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5161857
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160557, upper bound: 2.5156873
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160527, upper bound: 2.5156873
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5168130, upper bound: 2.5166823
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5168419
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5153302, upper bound: 2.5156948
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5153302, upper bound: 2.5153302
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5144801, upper bound: 2.5145783
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5144801, upper bound: 2.5145783
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166078
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5153302, upper bound: 2.5153302
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157314, upper bound: 2.5153302
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5168596, upper bound: 2.5166823
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166824, upper bound: 2.5166823
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.38 seconds

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

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5168598, upper bound: 2.5166823
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5166823
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166496
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5144801, upper bound: 2.5150638
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5144801, upper bound: 2.5150638
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5166823
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5166823
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5153302, upper bound: 2.5157441
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5156071, upper bound: 2.5157127
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5153302, upper bound: 2.5153302
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157212, upper bound: 2.5155667
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5166823
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5168645
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145715, upper bound: 2.5144801
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145715, upper bound: 2.5144801
time: 0.41 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 2.19 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166317
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160254, upper bound: 2.5160178
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5161097, upper bound: 2.5156873
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160979, upper bound: 2.5156873
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5168390, upper bound: 2.5166868
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5168428
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5145783, upper bound: 2.5144801
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5150510, upper bound: 2.5144801
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5144801, upper bound: 2.5150809
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5144801, upper bound: 2.5150809
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5166824
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5168596
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160522, upper bound: 2.5156873
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160504, upper bound: 2.5156873
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5161535
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5166493, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5161789
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5144801, upper bound: 2.5144801
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5144801, upper bound: 2.5144801
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5153302, upper bound: 2.5156261
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5153302, upper bound: 2.5153302
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5156873, upper bound: 2.5160138
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5156873, upper bound: 2.5157136
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5161857
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160557, upper bound: 2.5156873
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160527, upper bound: 2.5156873
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5168130, upper bound: 2.5166823
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5168419
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5153302, upper bound: 2.5156948
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5153302, upper bound: 2.5153302
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5144801, upper bound: 2.5145783
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5144801, upper bound: 2.5145783
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166078
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5153302, upper bound: 2.5153302
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5157314, upper bound: 2.5153302
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5168596, upper bound: 2.5166823
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5166824, upper bound: 2.5166823
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5168598, upper bound: 2.5166823
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5166823
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5166496
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5144801, upper bound: 2.5150638
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5144801, upper bound: 2.5150638
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5160178, upper bound: 2.5160178
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5166823
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5166823
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5153302, upper bound: 2.5157441
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5156071, upper bound: 2.5157127
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5153302, upper bound: 2.5153302
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5157212, upper bound: 2.5155667
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5166823
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5166823, upper bound: 2.5168645
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5145715, upper bound: 2.5144801
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 2.19
Output dim: 0, lower bound: -2.5145715, upper bound: 2.5144801

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142163
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142147
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5145803
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5153184, upper bound: 2.5145803
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 36

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143158, upper bound: 2.5143158
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145740, upper bound: 2.5143158
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5158258, upper bound: 2.5154472
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5155928, upper bound: 2.5154472
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5153479, upper bound: 2.5151770
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5151192, upper bound: 2.5151192
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142433, upper bound: 2.5143792
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142433, upper bound: 2.5143770
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5146073, upper bound: 2.5145803
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5153451, upper bound: 2.5145803
time: 0.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142433, upper bound: 2.5147324
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142433, upper bound: 2.5147324
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5164288
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5136667, upper bound: 2.5134940
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5136667, upper bound: 2.5134940
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
time: 0.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5148777
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5145803
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139528, upper bound: 2.5138278
time: 0.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5139528, upper bound: 2.5138278
time: 0.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5148739, upper bound: 2.5138278
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5148739, upper bound: 2.5138278
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5149086
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5145803
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5126738, upper bound: 2.5129011
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5126738, upper bound: 2.5129011
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5145803
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5145803
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143158, upper bound: 2.5147074
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143158, upper bound: 2.5143158
time: 0.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5154472, upper bound: 2.5154472
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5154472, upper bound: 2.5155058
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5136572, upper bound: 2.5134940
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5134940, upper bound: 2.5134940
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5158323, upper bound: 2.5154472
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5155613, upper bound: 2.5154472
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5159476
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157432, upper bound: 2.5157266
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5154472, upper bound: 2.5158324
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5154472, upper bound: 2.5158343
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5153505
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5145803
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143158, upper bound: 2.5143158
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143158, upper bound: 2.5143158
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142205
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142202
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5145803
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5145803
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5126738, upper bound: 2.5126738
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5126738, upper bound: 2.5126738
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143242, upper bound: 2.5143158
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147074, upper bound: 2.5143158
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5154472, upper bound: 2.5154472
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5158730, upper bound: 2.5154472
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 36

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5151192, upper bound: 2.5151192
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5151758, upper bound: 2.5151192
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142807, upper bound: 2.5138278
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5158657, upper bound: 2.5154472
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5158730, upper bound: 2.5154472
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5151192, upper bound: 2.5151210
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5151747, upper bound: 2.5151192
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148739
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148739
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5151192, upper bound: 2.5151696
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5151192, upper bound: 2.5155429
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143158, upper bound: 2.5143158
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5143158, upper bound: 2.5146899
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5126738, upper bound: 2.5126738
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5126738, upper bound: 2.5126738
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5150124
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5146636
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5151192, upper bound: 2.5151676
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5151192, upper bound: 2.5151192
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5154472, upper bound: 2.5158777
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5154472, upper bound: 2.5158896
time: 0.35 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.32 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142163
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142147
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5145803
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5153184, upper bound: 2.5145803
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5143158, upper bound: 2.5143158
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5145740, upper bound: 2.5143158
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5158258, upper bound: 2.5154472
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5155928, upper bound: 2.5154472
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5153479, upper bound: 2.5151770
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5151192, upper bound: 2.5151192
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5142433, upper bound: 2.5143792
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5142433, upper bound: 2.5143770
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5146073, upper bound: 2.5145803
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5153451, upper bound: 2.5145803
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5142433, upper bound: 2.5147324
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5142433, upper bound: 2.5147324
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5164288
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5136667, upper bound: 2.5134940
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5136667, upper bound: 2.5134940
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5148777
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5145803
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5139528, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5139528, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5148739, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5148739, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5149086
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5145803
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5126738, upper bound: 2.5129011
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5126738, upper bound: 2.5129011
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5145803
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5145803
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5143158, upper bound: 2.5147074
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5143158, upper bound: 2.5143158
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5154472, upper bound: 2.5154472
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5154472, upper bound: 2.5155058
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5136572, upper bound: 2.5134940
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5134940, upper bound: 2.5134940
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5158323, upper bound: 2.5154472
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5155613, upper bound: 2.5154472
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5159476
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5157432, upper bound: 2.5157266
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5154472, upper bound: 2.5158324
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5154472, upper bound: 2.5158343
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5153505
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5145803
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5143158, upper bound: 2.5143158
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5143158, upper bound: 2.5143158
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142205
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5142202
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5145803
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5145803
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5126738, upper bound: 2.5126738
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5126738, upper bound: 2.5126738
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5143242, upper bound: 2.5143158
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5147074, upper bound: 2.5143158
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5154472, upper bound: 2.5154472
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5158730, upper bound: 2.5154472
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5151192, upper bound: 2.5151192
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5151758, upper bound: 2.5151192
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5138278
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5142807, upper bound: 2.5138278
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5158657, upper bound: 2.5154472
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5158730, upper bound: 2.5154472
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5151192, upper bound: 2.5151210
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5151747, upper bound: 2.5151192
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148739
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5138278, upper bound: 2.5148739
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5149798, upper bound: 2.5149798
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5157266, upper bound: 2.5157266
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5151192, upper bound: 2.5151696
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5151192, upper bound: 2.5155429
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5143158, upper bound: 2.5143158
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5143158, upper bound: 2.5146899
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5126738, upper bound: 2.5126738
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5126738, upper bound: 2.5126738
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5150124
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5145803, upper bound: 2.5146636
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5151192, upper bound: 2.5151676
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5151192, upper bound: 2.5151192
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5154472, upper bound: 2.5158777
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.32
Output dim: 0, lower bound: -2.5154472, upper bound: 2.5158896

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142179, upper bound: 2.5142179
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145110, upper bound: 2.5142179
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5154798, upper bound: 2.5147367
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5141928
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5140715
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142179, upper bound: 2.5142179
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142179, upper bound: 2.5142179
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 36

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5154671
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5148065
time: 0.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 44

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 36

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5140715
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5140715
time: 0.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5141160
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5140715
time: 0.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5149558, upper bound: 2.5147367
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5134574, upper bound: 2.5135122
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5134574, upper bound: 2.5134651
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5144388, upper bound: 2.5142179
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142179, upper bound: 2.5142179
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5145352
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5140715
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 23

### Candidate
type: DSZ, layer: 3, pos: 44

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5132177, upper bound: 2.5134526
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5132177, upper bound: 2.5134508
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5122784, upper bound: 2.5125767
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5122784, upper bound: 2.5122784
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5140715
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5145746, upper bound: 2.5140715
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5154621, upper bound: 2.5147367
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5140715
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5140715
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5134574, upper bound: 2.5135833
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5134574, upper bound: 2.5135833
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5134574, upper bound: 2.5134574
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5134574, upper bound: 2.5134574
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Candidate
type: DSZ, layer: 3, pos: 49

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142179, upper bound: 2.5142179
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142179, upper bound: 2.5142179
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5134574, upper bound: 2.5134574
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5134574, upper bound: 2.5134574
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 0

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 42

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5123829, upper bound: 2.5132276
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5123829, upper bound: 2.5132276
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5154798
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5145553
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5144485
time: 0.37 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.07 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5142179, upper bound: 2.5142179
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5145110, upper bound: 2.5142179
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5154798, upper bound: 2.5147367
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5141928
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5140715
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5142179, upper bound: 2.5142179
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5142179, upper bound: 2.5142179
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5154671
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5148065
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5140715
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5140715
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5141160
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5140715
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5149558, upper bound: 2.5147367
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5134574, upper bound: 2.5135122
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5134574, upper bound: 2.5134651
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5144388, upper bound: 2.5142179
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5142179, upper bound: 2.5142179
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5145352
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5140715
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5132177, upper bound: 2.5134526
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5132177, upper bound: 2.5134508
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5122784, upper bound: 2.5125767
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5122784, upper bound: 2.5122784
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5140715
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5145746, upper bound: 2.5140715
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5154621, upper bound: 2.5147367
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5140715
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5140715
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5134574, upper bound: 2.5135833
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5134574, upper bound: 2.5135833
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5134574, upper bound: 2.5134574
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5134574, upper bound: 2.5134574
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5142179, upper bound: 2.5142179
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5142179, upper bound: 2.5142179
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5134574, upper bound: 2.5134574
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5134574, upper bound: 2.5134574
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5123829, upper bound: 2.5132276
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5123829, upper bound: 2.5132276
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5154798
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5147367, upper bound: 2.5147367
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5145553
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 2.07
Output dim: 0, lower bound: -2.5140715, upper bound: 2.5144485

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Candidate
type: DSZ, layer: 3, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 44

### Candidate
type: DSZ, layer: 3, pos: 36

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5137318, upper bound: 2.5127056
time: 0.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5137318, upper bound: 2.5127056
time: 0.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 1
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5134581, upper bound: 2.5142078
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5134581, upper bound: 2.5134581
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 23
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5134581, upper bound: 2.5134581
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5142074, upper bound: 2.5134581
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.2401047, 1.8555624, -1.2401047, 1.8555624, -3.0956669, 3.0956671
1: -1.4851047, 2.0027251, -1.4851047, 2.0027251, -3.4878297, 3.4878297
2: -1.4969569, 1.8678157, -1.4969569, 1.8678157, -3.3647728, 3.3647728
3: -1.7819047, 2.4911361, -1.7819047, 2.4911361, -4.2730408, 4.2730408
4: -2.2745841, 2.7287276, -2.2745841, 2.7287276, -5.0033116, 5.0033116

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 44
type: DSZ, layer: 3, pos: 26
type: DSZ, layer: 3, pos: 0
type: DSZ, layer: 3, pos: 13
type: DSZ, layer: 3, pos: 40
type: DSZ, layer: 3, pos: 42
type: DSZ, layer: 3, pos: 49
type: DSZ, layer: 3, pos: 36
type: DSZ, layer: 3, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 44

### Candidate
type: DSZ, layer: 3, pos: 26

### Candidate
type: DSZ, layer: 3, pos: 0

### Candidate
type: DSZ, layer: 3, pos: 13

### Candidate
type: DSZ, layer: 3, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5127056, upper bound: 2.5137318
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.5127056, upper bound: 2.5137318
time: 0.34 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 1.90 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.90
Output dim: 0, lower bound: -2.5137318, upper bound: 2.5127056
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.90
Output dim: 0, lower bound: -2.5137318, upper bound: 2.5127056
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.90
Output dim: 0, lower bound: -2.5134581, upper bound: 2.5142078
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.90
Output dim: 0, lower bound: -2.5134581, upper bound: 2.5134581
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.90
Output dim: 0, lower bound: -2.5134581, upper bound: 2.5134581
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.90
Output dim: 0, lower bound: -2.5142074, upper bound: 2.5134581
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 1.90
Output dim: 0, lower bound: -2.5127056, upper bound: 2.5137318
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 1.90
Output dim: 0, lower bound: -2.5127056, upper bound: 2.5137318

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.14 + 313.57 = 315.72 seconds
