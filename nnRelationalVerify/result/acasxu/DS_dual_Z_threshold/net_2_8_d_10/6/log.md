## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 8119.054186346224


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250)
1: (-2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836)
2: (-4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766)
3: (-5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625)
4: (-4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.73 + 2.18 = 4.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -8119.1353775, upper bound: 8119.1353776

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 41

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349762, upper bound: 8119.1349752
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349758, upper bound: 8119.1349758
time: 0.80 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.98 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.98
Output dim: 3, lower bound: -8119.1349762, upper bound: 8119.1349752
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.98
Output dim: 3, lower bound: -8119.1349758, upper bound: 8119.1349758

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349128, upper bound: 8119.1349131
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349130, upper bound: 8119.1349127
time: 0.83 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349131, upper bound: 8119.1349126
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349131, upper bound: 8119.1349127
time: 0.69 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.13 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -8119.1349128, upper bound: 8119.1349131
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -8119.1349130, upper bound: 8119.1349127
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -8119.1349131, upper bound: 8119.1349126
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.13
Output dim: 3, lower bound: -8119.1349131, upper bound: 8119.1349127

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349131, upper bound: 8119.1349125
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349130, upper bound: 8119.1349120
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349133, upper bound: 8119.1349125
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349132, upper bound: 8119.1349126
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349134, upper bound: 8119.1349127
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349131, upper bound: 8119.1349129
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349132, upper bound: 8119.1349132
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1349130, upper bound: 8119.1349126
time: 0.82 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.85 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 3, lower bound: -8119.1349131, upper bound: 8119.1349125
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 3, lower bound: -8119.1349130, upper bound: 8119.1349120
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 3, lower bound: -8119.1349133, upper bound: 8119.1349125
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 3, lower bound: -8119.1349132, upper bound: 8119.1349126
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 3, lower bound: -8119.1349134, upper bound: 8119.1349127
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 3, lower bound: -8119.1349131, upper bound: 8119.1349129
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 3, lower bound: -8119.1349132, upper bound: 8119.1349132
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.85
Output dim: 3, lower bound: -8119.1349130, upper bound: 8119.1349126

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347960, upper bound: 8119.1347951
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347959, upper bound: 8119.1347947
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347962, upper bound: 8119.1347952
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347961, upper bound: 8119.1347953
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347963, upper bound: 8119.1347952
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347957, upper bound: 8119.1347944
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347956, upper bound: 8119.1347943
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347961, upper bound: 8119.1347955
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347960, upper bound: 8119.1347953
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347961, upper bound: 8119.1347961
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347949, upper bound: 8119.1347950
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347958, upper bound: 8119.1347953
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347955, upper bound: 8119.1347950
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347960, upper bound: 8119.1347950
time: 1.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 22

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347949, upper bound: 8119.1347957
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1347961, upper bound: 8119.1347949
time: 0.91 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.51 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.51
Output dim: 3, lower bound: -8119.1347960, upper bound: 8119.1347951
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.51
Output dim: 3, lower bound: -8119.1347959, upper bound: 8119.1347947
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.51
Output dim: 3, lower bound: -8119.1347962, upper bound: 8119.1347952
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.51
Output dim: 3, lower bound: -8119.1347961, upper bound: 8119.1347953
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.51
Output dim: 3, lower bound: -8119.1347963, upper bound: 8119.1347952
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.51
Output dim: 3, lower bound: -8119.1347957, upper bound: 8119.1347944
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.51
Output dim: 3, lower bound: -8119.1347956, upper bound: 8119.1347943
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.51
Output dim: 3, lower bound: -8119.1347961, upper bound: 8119.1347955
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.51
Output dim: 3, lower bound: -8119.1347960, upper bound: 8119.1347953
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.51
Output dim: 3, lower bound: -8119.1347961, upper bound: 8119.1347961
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.51
Output dim: 3, lower bound: -8119.1347949, upper bound: 8119.1347950
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.51
Output dim: 3, lower bound: -8119.1347958, upper bound: 8119.1347953
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.51
Output dim: 3, lower bound: -8119.1347955, upper bound: 8119.1347950
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.51
Output dim: 3, lower bound: -8119.1347960, upper bound: 8119.1347950
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.51
Output dim: 3, lower bound: -8119.1347949, upper bound: 8119.1347957
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.51
Output dim: 3, lower bound: -8119.1347961, upper bound: 8119.1347949

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346488
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346484
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346495, upper bound: 8119.1346487
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346495, upper bound: 8119.1346486
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346487
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346482
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346493
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346489
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346490
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346496, upper bound: 8119.1346492
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346499, upper bound: 8119.1346486
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346499, upper bound: 8119.1346489
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346484
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346492
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346488
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346487
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346493
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346496, upper bound: 8119.1346491
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346496, upper bound: 8119.1346492
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346495, upper bound: 8119.1346494
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346495, upper bound: 8119.1346491
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346496, upper bound: 8119.1346493
time: 0.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346495
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346496, upper bound: 8119.1346495
time: 0.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346486
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346494, upper bound: 8119.1346490
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346492, upper bound: 8119.1346486
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346496, upper bound: 8119.1346491
time: 0.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346493
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346488
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346491
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346482
time: 0.83 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.56 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346488
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346484
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346495, upper bound: 8119.1346487
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346495, upper bound: 8119.1346486
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346487
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346482
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346493
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346489
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346490
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346496, upper bound: 8119.1346492
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346499, upper bound: 8119.1346486
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346499, upper bound: 8119.1346489
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346484
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346492
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346488
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346487
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346493
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346496, upper bound: 8119.1346491
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346496, upper bound: 8119.1346492
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346495, upper bound: 8119.1346494
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346495, upper bound: 8119.1346491
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346496, upper bound: 8119.1346493
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346495
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346496, upper bound: 8119.1346495
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346486
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346494, upper bound: 8119.1346490
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346492, upper bound: 8119.1346486
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346496, upper bound: 8119.1346491
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346493
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346498, upper bound: 8119.1346488
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346491
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -8119.1346497, upper bound: 8119.1346482

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339612
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339608
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339608
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339606
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339612
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339608
time: 0.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339611
time: 0.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339612
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339609
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339616
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339599
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339607
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339619, upper bound: 8119.1339611
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339604
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339610
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339618
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339614, upper bound: 8119.1339610
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339616
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339614, upper bound: 8119.1339618
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339614, upper bound: 8119.1339617
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339614, upper bound: 8119.1339613
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339607
time: 0.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339605
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339619
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339605
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339610
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339619
time: 0.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339617
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339609
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339608
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339608
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339619
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339619
time: 0.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339611
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339610
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339610
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339606
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339610
time: 0.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339614, upper bound: 8119.1339611
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339614, upper bound: 8119.1339608
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339618
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339609
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339621
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339609
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339618
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339618
time: 0.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339609
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339600
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339599
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339615
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339607
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339608
time: 0.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339610
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339612
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339614, upper bound: 8119.1339615
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339614, upper bound: 8119.1339601
time: 0.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339609
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339609
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339618
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339612
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339619
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339610
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339605
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339605
time: 0.94 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 5.37 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339612
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339608
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339608
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339606
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339612
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339608
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339611
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339612
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339609
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339616
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339599
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339607
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339619, upper bound: 8119.1339611
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339604
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339610
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339618
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339614, upper bound: 8119.1339610
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339616
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339614, upper bound: 8119.1339618
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339614, upper bound: 8119.1339617
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339614, upper bound: 8119.1339613
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339607
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339605
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339619
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339605
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339610
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339619
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339617
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339609
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339608
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339619
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339619
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339611
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339610
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339610
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339606
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339610
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339614, upper bound: 8119.1339611
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339614, upper bound: 8119.1339608
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339618
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339609
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339621
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339609
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339618
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339618
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339609
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339600
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339599
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339615
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339607
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339608
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339610
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339612
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339614, upper bound: 8119.1339615
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339614, upper bound: 8119.1339601
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339609
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339609
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339618
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339612
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339619
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339610
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339613, upper bound: 8119.1339605
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.37
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339605

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339611
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339618
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339604
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339610
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339611
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339611
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339604
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339612
time: 0.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339611
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339611
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339613
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339611
time: 0.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339615, upper bound: 8119.1339608
time: 0.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339612
time: 0.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339611
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339600
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339616
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339609
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339608
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339608
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339608
time: 0.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339616
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339602
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339605
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339607
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339615
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339604
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339609
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339606
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339609
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339602
time: 0.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339610
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339602
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339616
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339616, upper bound: 8119.1339615
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339611
time: 0.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339620
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339604
time: 0.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339619, upper bound: 8119.1339602
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339612
time: 0.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Candidate
type: DSZ, layer: 1, pos: 21

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 15

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339617, upper bound: 8119.1339600
time: 0.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -8119.1339618, upper bound: 8119.1339607
time: 0.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -27181.9257812, 29878.5917969, -27181.9257812, 29878.5917969, -57060.5156250, 57060.5156250
1: -2987.5979004, 2083.3527832, -2987.5979004, 2083.3527832, -5070.9506836, 5070.9506836
2: -4754.9384766, 5560.6879883, -4754.9384766, 5560.6879883, -10315.6259766, 10315.6259766
3: -5439.2436523, 3512.7326660, -5439.2436523, 3512.7326660, -8951.9765625, 8951.9765625
4: -4102.1416016, 4515.8403320, -4102.1416016, 4515.8403320, -8617.9824219, 8617.9824219

Time for backsubstitution: 2.77 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.91 + 415.42 = 420.33 seconds
