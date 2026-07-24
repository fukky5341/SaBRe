## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 9159.82088535655


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188)
1: (-5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188)
2: (-7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891)
3: (-2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812)
4: (-8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.78 + 2.09 = 2.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -9164.4030869, upper bound: 9164.4030869

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7845708, upper bound: 9163.7845708
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7845708, upper bound: 9163.7845708
time: 0.67 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.33 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 0, lower bound: -9163.7845708, upper bound: 9163.7845708
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 0, lower bound: -9163.7845708, upper bound: 9163.7845708

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7845708, upper bound: 9163.7845454
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7845454, upper bound: 9163.7845708
time: 0.82 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.5795386, upper bound: 9163.5782542
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.5782542, upper bound: 9163.5795386
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.16 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -9163.7845708, upper bound: 9163.7845454
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -9163.7845454, upper bound: 9163.7845708
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -9163.5795386, upper bound: 9163.5782542
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -9163.5782542, upper bound: 9163.5795386

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7219307, upper bound: 9162.7219307
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.7220220, upper bound: 9162.7219307
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7662701, upper bound: 9163.7636881
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.7651813, upper bound: 9163.7662938
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0986893, upper bound: 9162.0979590
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0979590, upper bound: 9162.0979590
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4755802, upper bound: 9163.4768293
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4755802, upper bound: 9163.4767263
time: 0.69 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.65 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -9162.7219307, upper bound: 9162.7219307
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -9162.7220220, upper bound: 9162.7219307
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -9163.7662701, upper bound: 9163.7636881
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -9163.7651813, upper bound: 9163.7662938
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -9162.0986893, upper bound: 9162.0979590
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -9162.0979590, upper bound: 9162.0979590
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -9163.4755802, upper bound: 9163.4768293
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -9163.4755802, upper bound: 9163.4767263

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3095555, upper bound: 9161.3093245
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3094449, upper bound: 9161.3094488
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0741333, upper bound: 9160.0736049
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0741333, upper bound: 9160.0740381
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6290083, upper bound: 9163.6258244
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6272424, upper bound: 9163.6258244
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6911800, upper bound: 9162.6916635
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.6911800, upper bound: 9162.6934358
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8512920, upper bound: 9161.8517206
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8511221, upper bound: 9161.8518291
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0979391, upper bound: 9162.0979391
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0979391, upper bound: 9162.0979391
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5430316, upper bound: 9162.5425368
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5430442, upper bound: 9162.5425052
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3635107, upper bound: 9163.3614159
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3634790, upper bound: 9163.3644467
time: 0.72 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.29 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -9161.3095555, upper bound: 9161.3093245
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -9161.3094449, upper bound: 9161.3094488
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -9160.0741333, upper bound: 9160.0736049
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -9160.0741333, upper bound: 9160.0740381
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -9163.6290083, upper bound: 9163.6258244
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -9163.6272424, upper bound: 9163.6258244
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -9162.6911800, upper bound: 9162.6916635
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -9162.6911800, upper bound: 9162.6934358
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -9161.8512920, upper bound: 9161.8517206
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -9161.8511221, upper bound: 9161.8518291
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -9162.0979391, upper bound: 9162.0979391
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -9162.0979391, upper bound: 9162.0979391
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -9162.5430316, upper bound: 9162.5425368
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -9162.5430442, upper bound: 9162.5425052
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -9163.3635107, upper bound: 9163.3614159
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.29
Output dim: 0, lower bound: -9163.3634790, upper bound: 9163.3644467

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.0928779, upper bound: 9157.0928779
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.0928779, upper bound: 9157.0928779
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.0928779, upper bound: 9157.0928779
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9157.0928779, upper bound: 9157.0928779
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0729935, upper bound: 9160.0725515
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0725515, upper bound: 9160.0725515
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0741333, upper bound: 9160.0736049
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.0736049, upper bound: 9160.0740381
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6272117, upper bound: 9163.6256884
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6256884, upper bound: 9163.6256884
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6272308, upper bound: 9163.6258244
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6258244, upper bound: 9163.6258244
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6479462, upper bound: 9161.6482202
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6479462, upper bound: 9161.6482202
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7920200, upper bound: 9161.7928636
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7920200, upper bound: 9161.7928636
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7573146, upper bound: 9161.7568250
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7568250, upper bound: 9161.7576879
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8510962, upper bound: 9161.8510962
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8510962, upper bound: 9161.8518069
time: 0.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8510962, upper bound: 9161.8510962
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8510962, upper bound: 9161.8510962
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9791680, upper bound: 9161.9791680
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9791680, upper bound: 9161.9791680
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3598391, upper bound: 9162.3607258
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3595988, upper bound: 9162.3595988
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1911621, upper bound: 9162.1911621
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1916622, upper bound: 9162.1911621
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.2854133, upper bound: 9163.2836218
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.2836218, upper bound: 9163.2836218
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9330567, upper bound: 9162.9330567
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9330567, upper bound: 9162.9330567
time: 0.64 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.16 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -9157.0928779, upper bound: 9157.0928779
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -9157.0928779, upper bound: 9157.0928779
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -9157.0928779, upper bound: 9157.0928779
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.16
Output dim: 0, lower bound: -9157.0928779, upper bound: 9157.0928779
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9160.0729935, upper bound: 9160.0725515
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9160.0725515, upper bound: 9160.0725515
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9160.0741333, upper bound: 9160.0736049
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9160.0736049, upper bound: 9160.0740381
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9163.6272117, upper bound: 9163.6256884
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9163.6256884, upper bound: 9163.6256884
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9163.6272308, upper bound: 9163.6258244
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9163.6258244, upper bound: 9163.6258244
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9161.6479462, upper bound: 9161.6482202
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9161.6479462, upper bound: 9161.6482202
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9161.7920200, upper bound: 9161.7928636
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9161.7920200, upper bound: 9161.7928636
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9161.7573146, upper bound: 9161.7568250
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9161.7568250, upper bound: 9161.7576879
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9161.8510962, upper bound: 9161.8510962
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9161.8510962, upper bound: 9161.8518069
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9161.8510962, upper bound: 9161.8510962
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9161.8510962, upper bound: 9161.8510962
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9161.9791680, upper bound: 9161.9791680
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9161.9791680, upper bound: 9161.9791680
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9162.3598391, upper bound: 9162.3607258
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9162.3595988, upper bound: 9162.3595988
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9162.1911621, upper bound: 9162.1911621
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9162.1916622, upper bound: 9162.1911621
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9163.2854133, upper bound: 9163.2836218
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9163.2836218, upper bound: 9163.2836218
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9162.9330567, upper bound: 9162.9330567
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.16
Output dim: 0, lower bound: -9162.9330567, upper bound: 9162.9330567

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.2438298, upper bound: 9159.2437690
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.2438298, upper bound: 9159.2437690
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.7295144, upper bound: 9158.7295144
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.7295144, upper bound: 9158.7295144
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.9703240, upper bound: 9159.9700904
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.9700904, upper bound: 9159.9700904
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.8431060, upper bound: 9159.8436259
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.8431060, upper bound: 9159.8431844
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6285718, upper bound: 9163.6252054
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.6252054, upper bound: 9163.6252054
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.5995481, upper bound: 9163.6002952
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.5995481, upper bound: 9163.6004809
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8772308, upper bound: 9162.8759262
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8759262, upper bound: 9162.8759262
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4222512, upper bound: 9163.4222512
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4222512, upper bound: 9163.4222512
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6478993, upper bound: 9161.6479886
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6478993, upper bound: 9161.6481690
time: 0.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6472986, upper bound: 9161.6472986
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6472986, upper bound: 9161.6475822
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7920200, upper bound: 9161.7920200
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7920200, upper bound: 9161.7928636
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6990941, upper bound: 9161.6998486
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6990941, upper bound: 9161.6998720
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7573146, upper bound: 9161.7568250
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7569013, upper bound: 9161.7568250
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7568041, upper bound: 9161.7568041
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7568041, upper bound: 9161.7576699
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7892867, upper bound: 9161.7892867
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7892867, upper bound: 9161.7892867
time: 0.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6990454, upper bound: 9161.6995269
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6990454, upper bound: 9161.6995269
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8510962, upper bound: 9161.8510962
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8510962, upper bound: 9161.8510962
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5963072, upper bound: 9161.5963072
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5963072, upper bound: 9161.5963072
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9785169, upper bound: 9161.9785169
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9785169, upper bound: 9161.9785169
time: 0.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9276945, upper bound: 9161.9284901
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9276945, upper bound: 9161.9284866
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9974410, upper bound: 9161.9980127
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9977361, upper bound: 9161.9980144
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9974410, upper bound: 9161.9974410
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9974410, upper bound: 9161.9974410
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0810837, upper bound: 9162.0810837
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0810837, upper bound: 9162.0810837
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1916622, upper bound: 9162.1911614
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1914882, upper bound: 9162.1911614
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0809944, upper bound: 9163.0802717
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0802717, upper bound: 9163.0802717
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.2808116, upper bound: 9163.2808116
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.2808116, upper bound: 9163.2808116
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9317784, upper bound: 9162.9317784
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9317784, upper bound: 9162.9317784
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.1549595, upper bound: 9158.1549595
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9158.1549595, upper bound: 9158.1549595
time: 0.76 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.10 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.10
Output dim: 0, lower bound: -9159.2438298, upper bound: 9159.2437690
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.10
Output dim: 0, lower bound: -9159.2438298, upper bound: 9159.2437690
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.10
Output dim: 0, lower bound: -9158.7295144, upper bound: 9158.7295144
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.10
Output dim: 0, lower bound: -9158.7295144, upper bound: 9158.7295144
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9159.9703240, upper bound: 9159.9700904
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9159.9700904, upper bound: 9159.9700904
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9159.8431060, upper bound: 9159.8436259
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9159.8431060, upper bound: 9159.8431844
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9163.6285718, upper bound: 9163.6252054
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9163.6252054, upper bound: 9163.6252054
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9163.5995481, upper bound: 9163.6002952
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9163.5995481, upper bound: 9163.6004809
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9162.8772308, upper bound: 9162.8759262
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9162.8759262, upper bound: 9162.8759262
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9163.4222512, upper bound: 9163.4222512
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9163.4222512, upper bound: 9163.4222512
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.6478993, upper bound: 9161.6479886
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.6478993, upper bound: 9161.6481690
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.6472986, upper bound: 9161.6472986
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.6472986, upper bound: 9161.6475822
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.7920200, upper bound: 9161.7920200
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.7920200, upper bound: 9161.7928636
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.6990941, upper bound: 9161.6998486
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.6990941, upper bound: 9161.6998720
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.7573146, upper bound: 9161.7568250
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.7569013, upper bound: 9161.7568250
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.7568041, upper bound: 9161.7568041
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.7568041, upper bound: 9161.7576699
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.7892867, upper bound: 9161.7892867
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.7892867, upper bound: 9161.7892867
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.6990454, upper bound: 9161.6995269
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.6990454, upper bound: 9161.6995269
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.8510962, upper bound: 9161.8510962
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.8510962, upper bound: 9161.8510962
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.5963072, upper bound: 9161.5963072
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.5963072, upper bound: 9161.5963072
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.9785169, upper bound: 9161.9785169
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.9785169, upper bound: 9161.9785169
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.9276945, upper bound: 9161.9284901
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.9276945, upper bound: 9161.9284866
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.9974410, upper bound: 9161.9980127
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.9977361, upper bound: 9161.9980144
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.9974410, upper bound: 9161.9974410
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9161.9974410, upper bound: 9161.9974410
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9162.0810837, upper bound: 9162.0810837
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9162.0810837, upper bound: 9162.0810837
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9162.1916622, upper bound: 9162.1911614
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9162.1914882, upper bound: 9162.1911614
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9163.0809944, upper bound: 9163.0802717
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9163.0802717, upper bound: 9163.0802717
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9163.2808116, upper bound: 9163.2808116
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9163.2808116, upper bound: 9163.2808116
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9162.9317784, upper bound: 9162.9317784
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -9162.9317784, upper bound: 9162.9317784
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.10
Output dim: 0, lower bound: -9158.1549595, upper bound: 9158.1549595
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.10
Output dim: 0, lower bound: -9158.1549595, upper bound: 9158.1549595

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.9701237, upper bound: 9159.9700904
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.9703240, upper bound: 9159.9700904
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7320492, upper bound: 9159.7320492
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.7320492, upper bound: 9159.7320492
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.5355952, upper bound: 9159.5355952
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.5355952, upper bound: 9159.5362173
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.3251883, upper bound: 9159.3251906
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9159.3251883, upper bound: 9159.3251883
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4247607, upper bound: 9163.4216332
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4234325, upper bound: 9163.4216332
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4063469, upper bound: 9163.4063469
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4063469, upper bound: 9163.4063469
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3806236, upper bound: 9163.3806236
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.3806236, upper bound: 9163.3806236
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.5990651, upper bound: 9163.5990651
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.5990651, upper bound: 9163.6000031
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7439264, upper bound: 9161.7434491
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7439264, upper bound: 9161.7434491
time: 0.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8757198, upper bound: 9162.8757198
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8757198, upper bound: 9162.8757198
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5871010, upper bound: 9162.5871010
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5871010, upper bound: 9162.5871010
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4752252, upper bound: 9161.4752252
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4752252, upper bound: 9161.4752252
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4502996, upper bound: 9161.4504201
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4502996, upper bound: 9161.4502996
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6477315, upper bound: 9161.6477315
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6477315, upper bound: 9161.6480070
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6471339, upper bound: 9161.6471339
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6471339, upper bound: 9161.6471339
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6471701, upper bound: 9161.6472989
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6471701, upper bound: 9161.6474579
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7920200, upper bound: 9161.7920200
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7920200, upper bound: 9161.7920200
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7919921, upper bound: 9161.7927902
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7919921, upper bound: 9161.7928382
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6983758, upper bound: 9161.6983975
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6983758, upper bound: 9161.6995137
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6990941, upper bound: 9161.6990941
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6990941, upper bound: 9161.6998720
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7420467, upper bound: 9161.7415652
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7417708, upper bound: 9161.7415652
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6957640, upper bound: 9161.6956642
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6957298, upper bound: 9161.6956642
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6840065, upper bound: 9161.6840065
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6840065, upper bound: 9161.6840065
time: 2.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5477557, upper bound: 9161.5481520
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5477557, upper bound: 9161.5481520
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5513758, upper bound: 9161.5513758
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5513758, upper bound: 9161.5513758
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6267900, upper bound: 9161.6267900
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6267900, upper bound: 9161.6267900
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6984177, upper bound: 9161.6984177
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6984177, upper bound: 9161.6989243
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5755927, upper bound: 9161.5755927
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5755927, upper bound: 9161.5761034
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8504173, upper bound: 9161.8504173
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8504173, upper bound: 9161.8504173
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7892867, upper bound: 9161.7892867
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7892867, upper bound: 9161.7892867
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5545125, upper bound: 9161.5545125
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5545125, upper bound: 9161.5545125
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5545125, upper bound: 9161.5545125
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5545125, upper bound: 9161.5545125
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9785127, upper bound: 9161.9785127
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9785127, upper bound: 9161.9785127
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9785169, upper bound: 9161.9785169
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9785169, upper bound: 9161.9785169
time: 0.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9062186, upper bound: 9161.9070408
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9062186, upper bound: 9161.9070415
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9276945, upper bound: 9161.9276945
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9276945, upper bound: 9161.9284866
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1952568, upper bound: 9160.1952568
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9160.1952568, upper bound: 9160.1952568
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9974410, upper bound: 9161.9980144
time: 0.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9977361, upper bound: 9161.9980133
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9051451, upper bound: 9161.9051451
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9051451, upper bound: 9161.9051451
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9974410, upper bound: 9161.9974410
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.9974410, upper bound: 9161.9974410
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0804615, upper bound: 9162.0804615
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.0804615, upper bound: 9162.0804615
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8037020, upper bound: 9161.8037020
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.8037020, upper bound: 9161.8037020
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1906218, upper bound: 9162.1905994
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1910240, upper bound: 9162.1905994
time: 0.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1914428, upper bound: 9162.1911256
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.1911256, upper bound: 9162.1911256
time: 0.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0809653, upper bound: 9163.0802593
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.0802593, upper bound: 9163.0802593
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5096214, upper bound: 9162.5096214
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5100511, upper bound: 9162.5096214
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6844141, upper bound: 9161.6844141
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6844141, upper bound: 9161.6844141
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3203624, upper bound: 9162.3203624
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.3203624, upper bound: 9162.3203624
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9317784, upper bound: 9162.9317784
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.9317784, upper bound: 9162.9317784
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8501243, upper bound: 9162.8501243
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.8501243, upper bound: 9162.8501243
time: 0.65 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 2.69 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9159.9701237, upper bound: 9159.9700904
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9159.9703240, upper bound: 9159.9700904
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -9159.7320492, upper bound: 9159.7320492
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -9159.7320492, upper bound: 9159.7320492
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -9159.5355952, upper bound: 9159.5355952
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -9159.5355952, upper bound: 9159.5362173
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -9159.3251883, upper bound: 9159.3251906
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 2.69
Output dim: 0, lower bound: -9159.3251883, upper bound: 9159.3251883
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9163.4247607, upper bound: 9163.4216332
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9163.4234325, upper bound: 9163.4216332
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9163.4063469, upper bound: 9163.4063469
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9163.4063469, upper bound: 9163.4063469
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9163.3806236, upper bound: 9163.3806236
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9163.3806236, upper bound: 9163.3806236
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9163.5990651, upper bound: 9163.5990651
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9163.5990651, upper bound: 9163.6000031
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.7439264, upper bound: 9161.7434491
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.7439264, upper bound: 9161.7434491
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.8757198, upper bound: 9162.8757198
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.8757198, upper bound: 9162.8757198
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.5871010, upper bound: 9162.5871010
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.5871010, upper bound: 9162.5871010
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.4752252, upper bound: 9161.4752252
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.4752252, upper bound: 9161.4752252
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.4502996, upper bound: 9161.4504201
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.4502996, upper bound: 9161.4502996
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6477315, upper bound: 9161.6477315
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6477315, upper bound: 9161.6480070
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6471339, upper bound: 9161.6471339
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6471339, upper bound: 9161.6471339
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6471701, upper bound: 9161.6472989
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6471701, upper bound: 9161.6474579
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.7920200, upper bound: 9161.7920200
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.7920200, upper bound: 9161.7920200
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.7919921, upper bound: 9161.7927902
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.7919921, upper bound: 9161.7928382
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6983758, upper bound: 9161.6983975
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6983758, upper bound: 9161.6995137
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6990941, upper bound: 9161.6990941
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6990941, upper bound: 9161.6998720
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.7420467, upper bound: 9161.7415652
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.7417708, upper bound: 9161.7415652
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6957640, upper bound: 9161.6956642
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6957298, upper bound: 9161.6956642
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6840065, upper bound: 9161.6840065
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6840065, upper bound: 9161.6840065
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.5477557, upper bound: 9161.5481520
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.5477557, upper bound: 9161.5481520
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.5513758, upper bound: 9161.5513758
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.5513758, upper bound: 9161.5513758
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6267900, upper bound: 9161.6267900
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6267900, upper bound: 9161.6267900
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6984177, upper bound: 9161.6984177
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6984177, upper bound: 9161.6989243
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.5755927, upper bound: 9161.5755927
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.5755927, upper bound: 9161.5761034
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.8504173, upper bound: 9161.8504173
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.8504173, upper bound: 9161.8504173
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.7892867, upper bound: 9161.7892867
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.7892867, upper bound: 9161.7892867
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.5545125, upper bound: 9161.5545125
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.5545125, upper bound: 9161.5545125
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.5545125, upper bound: 9161.5545125
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.5545125, upper bound: 9161.5545125
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.9785127, upper bound: 9161.9785127
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.9785127, upper bound: 9161.9785127
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.9785169, upper bound: 9161.9785169
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.9785169, upper bound: 9161.9785169
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.9062186, upper bound: 9161.9070408
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.9062186, upper bound: 9161.9070415
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.9276945, upper bound: 9161.9276945
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.9276945, upper bound: 9161.9284866
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9160.1952568, upper bound: 9160.1952568
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9160.1952568, upper bound: 9160.1952568
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.9974410, upper bound: 9161.9980144
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.9977361, upper bound: 9161.9980133
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.9051451, upper bound: 9161.9051451
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.9051451, upper bound: 9161.9051451
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.9974410, upper bound: 9161.9974410
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.9974410, upper bound: 9161.9974410
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.0804615, upper bound: 9162.0804615
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.0804615, upper bound: 9162.0804615
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.8037020, upper bound: 9161.8037020
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.8037020, upper bound: 9161.8037020
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.1906218, upper bound: 9162.1905994
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.1910240, upper bound: 9162.1905994
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.1914428, upper bound: 9162.1911256
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.1911256, upper bound: 9162.1911256
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9163.0809653, upper bound: 9163.0802593
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9163.0802593, upper bound: 9163.0802593
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.5096214, upper bound: 9162.5096214
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.5100511, upper bound: 9162.5096214
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6844141, upper bound: 9161.6844141
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9161.6844141, upper bound: 9161.6844141
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.3203624, upper bound: 9162.3203624
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.3203624, upper bound: 9162.3203624
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.9317784, upper bound: 9162.9317784
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.9317784, upper bound: 9162.9317784
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.8501243, upper bound: 9162.8501243
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 2.69
Output dim: 0, lower bound: -9162.8501243, upper bound: 9162.8501243

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 47

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.9700500, upper bound: 9159.9700338
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9159.9700338, upper bound: 9159.9700338
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -9156.3559197, upper bound: 9156.3559197
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -9156.3559197, upper bound: 9156.3559197
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4747817, upper bound: 9161.4745099
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4745099, upper bound: 9161.4745099
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5263959, upper bound: 9162.5249426
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5249426, upper bound: 9162.5249426
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4063469, upper bound: 9163.4063469
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.4063469, upper bound: 9163.4063469
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.2860532, upper bound: 9163.2860532
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.2860532, upper bound: 9163.2860532
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4890601, upper bound: 9162.4892377
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4890601, upper bound: 9162.4895066
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.2012513, upper bound: 9163.2012513
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.2012513, upper bound: 9163.2012513
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5884541, upper bound: 9162.5884541
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5884541, upper bound: 9162.5884541
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.5273385, upper bound: 9163.5273385
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9163.5273385, upper bound: 9163.5273385
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5893883, upper bound: 9161.5893883
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5898618, upper bound: 9161.5893883
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6881150, upper bound: 9161.6873962
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6873962, upper bound: 9161.6873962
time: 0.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5721899, upper bound: 9161.5721899
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5721899, upper bound: 9161.5721899
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5721899, upper bound: 9161.5721899
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5721899, upper bound: 9161.5721899
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4600553, upper bound: 9162.4600553
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.4600553, upper bound: 9162.4600553
time: 0.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5856016, upper bound: 9162.5856016
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9162.5856016, upper bound: 9162.5856016
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4751861, upper bound: 9161.4751861
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4751861, upper bound: 9161.4751861
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4752252, upper bound: 9161.4752252
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4752252, upper bound: 9161.4752252
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 9

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4501556, upper bound: 9161.4501895
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4501556, upper bound: 9161.4502570
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4502135, upper bound: 9161.4502135
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.4502135, upper bound: 9161.4502135
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5452134, upper bound: 9161.5452134
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5452134, upper bound: 9161.5452134
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5171992, upper bound: 9161.5174633
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5171992, upper bound: 9161.5171992
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5368675, upper bound: 9161.5368675
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5368675, upper bound: 9161.5368675
time: 0.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6002369, upper bound: 9161.6002369
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6002369, upper bound: 9161.6002369
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6471701, upper bound: 9161.6471701
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6471701, upper bound: 9161.6472989
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5369393, upper bound: 9161.5371818
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5369393, upper bound: 9161.5369393
time: 0.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7920200, upper bound: 9161.7920200
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7920200, upper bound: 9161.7920200
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 44

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7914550, upper bound: 9161.7914550
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7914550, upper bound: 9161.7914550
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7593119, upper bound: 9161.7600084
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7593119, upper bound: 9161.7593119
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7862689, upper bound: 9161.7862689
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7862689, upper bound: 9161.7870952
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6978186, upper bound: 9161.6978186
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6978186, upper bound: 9161.6978581
time: 0.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6983758, upper bound: 9161.6983758
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6983758, upper bound: 9161.6995137
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6640157, upper bound: 9161.6640157
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6640157, upper bound: 9161.6640157
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5694103, upper bound: 9161.5701465
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5694103, upper bound: 9161.5696647
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6985399, upper bound: 9161.6980788
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6985435, upper bound: 9161.6980788
time: 0.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7410080, upper bound: 9161.7410080
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.7411543, upper bound: 9161.7410080
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6783832, upper bound: 9161.6783071
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6784061, upper bound: 9161.6783071
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6951070, upper bound: 9161.6951070
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6951070, upper bound: 9161.6951070
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6840065, upper bound: 9161.6840065
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6840065, upper bound: 9161.6840065
time: 0.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 39

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6493719, upper bound: 9161.6493719
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6493719, upper bound: 9161.6493719
time: 0.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5386694, upper bound: 9161.5390708
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5386694, upper bound: 9161.5386694
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3990899, upper bound: 9161.3993195
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.3990899, upper bound: 9161.3995604
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5139062, upper bound: 9161.5139062
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5139062, upper bound: 9161.5139062
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 17

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5153445, upper bound: 9161.5153445
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.5153445, upper bound: 9161.5153445
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6267900, upper bound: 9161.6267900
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6267900, upper bound: 9161.6267900
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 31

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6004281, upper bound: 9161.6004281
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6004281, upper bound: 9161.6004281
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6984177, upper bound: 9161.6984177
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6984177, upper bound: 9161.6984177
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6984177, upper bound: 9161.6985036
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -9161.6984177, upper bound: 9161.6989243
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6571.1416016, 4892.4965820, -6571.1416016, 4892.4965820, -11463.6367188, 11463.6367188
1: -5223.5683594, 4703.7553711, -5223.5683594, 4703.7553711, -9927.3242188, 9927.3242188
2: -7584.2861328, 5120.6601562, -7584.2861328, 5120.6601562, -12704.9462891, 12704.9462891
3: -2863.8518066, 7293.6987305, -2863.8518066, 7293.6987305, -10157.5498047, 10157.5507812
4: -8426.3330078, 5078.6298828, -8426.3330078, 5078.6298828, -13504.9628906, 13504.9628906

Time for backsubstitution: 0.96 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.87 + 417.25 = 420.11 seconds
