## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 398.85261092052


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-179.2202911, 303.3570251, -179.2202911, 303.3570251, -482.5773315, 482.5773315)
1: (-197.4927673, 268.0154114, -197.4927673, 268.0154114, -465.5081482, 465.5081482)
2: (-197.7517548, 272.0600586, -197.7517548, 272.0600586, -469.8117371, 469.8117371)
3: (-234.1109924, 308.6250000, -234.1109924, 308.6250000, -542.7359619, 542.7359619)
4: (-201.8509827, 312.5909424, -201.8509827, 312.5909424, -514.4418945, 514.4418945)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.63 + 2.58 = 3.21 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -398.9323974, upper bound: 398.9323973

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8849160, upper bound: 398.9180450
time: 0.85 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8848071, upper bound: 398.8848071
time: 0.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.65 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 0, lower bound: -398.8849160, upper bound: 398.9180450
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 0, lower bound: -398.8848071, upper bound: 398.8848071

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -179.2202911, 303.3570251, -164.8828735, 273.6366882, -452.8569946, 468.2398987
1: -197.4927673, 268.0154114, -181.5148926, 242.5971832, -440.0898438, 449.5303040
2: -197.7517548, 272.0600586, -181.4661713, 246.9006042, -444.6523438, 453.5261230
3: -234.1109924, 308.6250000, -214.6846161, 279.4453430, -513.5563354, 523.3095093
4: -201.8509827, 312.5909424, -184.8080750, 283.3780212, -485.2290039, 497.3990173

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8845823, upper bound: 398.8845823
time: 1.09 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8845823, upper bound: 398.8848071
time: 0.82 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -178.0285950, 301.3497620, -382.0293579, 576.7352295, -750.0909424, 670.6077881
1: -196.1907959, 266.3074036, -417.7243652, 527.1749878, -718.0180664, 672.2918701
2: -196.4221954, 270.3563843, -416.6148071, 536.4212646, -727.4710693, 677.1156006
3: -232.5894928, 306.6549072, -487.5961609, 608.3463135, -834.6773071, 785.5539551
4: -200.5400543, 310.5973816, -418.1431580, 614.5153198, -812.7661743, 720.4613037

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8848071, upper bound: 398.8845823
time: 0.91 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8848071, upper bound: 398.8848071
time: 1.00 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.54 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 0, lower bound: -398.8845823, upper bound: 398.8845823
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 0, lower bound: -398.8845823, upper bound: 398.8848071
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 0, lower bound: -398.8848071, upper bound: 398.8845823
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 0, lower bound: -398.8848071, upper bound: 398.8848071

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -164.8828735, 273.6366882, -164.8828735, 273.6366882, -438.5195618, 438.5195618
1: -181.5148926, 242.5971832, -181.5148926, 242.5971832, -424.1119995, 424.1119995
2: -181.4661713, 246.9006042, -181.4661713, 246.9006042, -428.3666992, 428.3666992
3: -214.6846161, 279.4453430, -214.6846161, 279.4453430, -494.1299438, 494.1299438
4: -184.8080750, 283.3780212, -184.8080750, 283.3780212, -468.1860962, 468.1860962

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8660263, upper bound: 398.9155136
time: 1.14 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835229, upper bound: 398.9171847
time: 0.67 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -381.9187012, 576.5624390, -164.8828735, 273.6366882, -642.8849487, 736.9779053
1: -417.5991211, 527.0137329, -181.5148926, 242.5971832, -648.6752319, 703.3931274
2: -416.4932556, 536.2599487, -181.4661713, 246.9006042, -654.1002197, 712.5161743
3: -487.4469604, 608.1581421, -214.6846161, 279.4453430, -758.2739868, 816.7816162
4: -418.0183105, 614.3297119, -184.8080750, 283.3780212, -693.2182007, 796.9906616

Time for backsubstitution: 0.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_A1

### Relational analysis result of NS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8660263, upper bound: 398.9155136
time: 0.90 seconds

## Relational analysis of NS_B1_A2_A2

### Relational analysis result of NS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835229, upper bound: 398.9171847
time: 0.92 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -164.8828735, 273.6366882, -381.9187012, 576.5624390, -736.9779053, 642.8848877
1: -181.5148926, 242.5971832, -417.5991211, 527.0137329, -703.3931274, 648.6752319
2: -181.4661713, 246.9006042, -416.4932556, 536.2599487, -712.5161743, 654.1002197
3: -214.6846161, 279.4453430, -487.4469604, 608.1581421, -816.7816162, 758.2739868
4: -184.8080750, 283.3780212, -418.0183105, 614.3297119, -796.9906616, 693.2182007

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829808, upper bound: 398.8658788
time: 0.79 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8829808, upper bound: 398.8831321
time: 0.84 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -382.1474609, 576.9548950, -382.1474609, 576.9548950, -937.3775635, 937.3775635
1: -417.8582153, 527.3723145, -417.8582153, 527.3723145, -923.8921509, 923.8921509
2: -416.7462769, 536.6198730, -416.7462769, 536.6198730, -933.7025757, 933.7026367
3: -487.7610474, 608.5759277, -487.7610474, 608.5759277, -1077.0296631, 1077.0296631
4: -418.2799072, 614.7428589, -418.2799072, 614.7428589, -1018.3641968, 1018.3641357

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8658788, upper bound: 398.8831380
time: 0.88 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8831321, upper bound: 398.8832429
time: 1.04 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.74 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -398.8660263, upper bound: 398.9155136
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -398.8835229, upper bound: 398.9171847
NS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -398.8660263, upper bound: 398.9155136
NS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -398.8835229, upper bound: 398.9171847
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -398.8829808, upper bound: 398.8658788
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -398.8829808, upper bound: 398.8831321
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -398.8658788, upper bound: 398.8831380
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 2.74
Output dim: 0, lower bound: -398.8831321, upper bound: 398.8832429

## BFS NS instance: NS_B1_A1_A1

### Backsubstitution after applying NS history:
0: -153.7317200, 252.2772827, -164.6252747, 273.1701050, -426.9017944, 416.9025269
1: -169.0001678, 223.6766815, -181.2254486, 242.1801758, -411.1803284, 404.9021301
2: -168.9764252, 227.4005737, -181.1805267, 246.4721680, -415.4486084, 408.5811157
3: -199.1547699, 257.6705627, -214.3279419, 278.9593811, -478.1141357, 471.9985046
4: -171.7021637, 261.3429871, -184.5051727, 282.8896484, -454.5917969, 445.8481445

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_A1_A1

### Relational analysis result of NS_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817404, upper bound: 398.9198913
time: 0.75 seconds

## Relational analysis of NS_B1_A1_A1_A2

### Relational analysis result of NS_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9007163, upper bound: 398.9281676
time: 1.10 seconds

## BFS NS instance: NS_B1_A1_A2

### Backsubstitution after applying NS history:
0: -163.3380432, 270.8208618, -164.8828735, 273.6366882, -436.9747314, 435.7037354
1: -179.7960052, 240.1278992, -181.5148926, 242.5971832, -422.3931274, 421.6427917
2: -179.7642059, 244.4613647, -181.4661713, 246.9006042, -426.6647949, 425.9274902
3: -212.6234283, 276.6451721, -214.6846161, 279.4453430, -492.0687561, 491.3297729
4: -183.0366974, 280.5787659, -184.8080750, 283.3780212, -466.4147339, 465.3868408

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9200177, upper bound: 398.9034343
time: 0.95 seconds

## Relational analysis of NS_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9285787, upper bound: 398.9285787
time: 0.78 seconds

## BFS NS instance: NS_B1_A2_A1

### Backsubstitution after applying NS history:
0: -371.2971497, 557.0361938, -164.6252747, 273.1701050, -631.2890625, 716.0309448
1: -405.6079407, 509.6666565, -181.2254486, 242.1801758, -635.9122314, 684.7954712
2: -404.5974121, 518.6414185, -181.1805267, 246.4721680, -641.2670288, 693.5394897
3: -472.7274780, 588.3479004, -214.3279419, 278.9593811, -742.7283936, 795.4498901
4: -405.3910217, 594.0910645, -184.5051727, 282.8896484, -679.6483765, 775.6606445

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8151986, upper bound: 398.9054748
time: 0.79 seconds

## Relational analysis of NS_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_A1_A1

### Relational analysis result of NS_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8637842, upper bound: 398.9137791
time: 1.03 seconds

## Relational analysis of NS_B1_A2_A1_A2

### Relational analysis result of NS_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8652542, upper bound: 398.9152031
time: 1.07 seconds

## BFS NS instance: NS_B1_A2_A2

### Backsubstitution after applying NS history:
0: -380.5660400, 573.9440308, -164.8828735, 273.6366882, -641.4078979, 734.3572388
1: -416.0835876, 524.8048096, -181.5148926, 242.5971832, -647.0747681, 701.1510620
2: -414.9791260, 533.9889526, -181.4661713, 246.9006042, -652.4486694, 710.2315063
3: -485.5814514, 605.6345215, -214.6846161, 279.4453430, -756.3324585, 814.2229614
4: -416.4398804, 611.7735596, -184.8080750, 283.3780212, -691.5787354, 794.4026489

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8351220, upper bound: 398.9055065
time: 0.75 seconds

## Relational analysis of NS_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_A2_A1

### Relational analysis result of NS_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8810653, upper bound: 398.9139819
time: 0.77 seconds

## Relational analysis of NS_B1_A2_A2_A2

### Relational analysis result of NS_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8835037, upper bound: 398.9168129
time: 1.01 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -164.6252747, 273.1701050, -371.2971191, 557.0361938, -716.0309448, 631.2890015
1: -181.2254486, 242.1801758, -405.6079407, 509.6666565, -684.7954712, 635.9121704
2: -181.1805267, 246.4721680, -404.5974121, 518.6414185, -693.5394897, 641.2669678
3: -214.3279419, 278.9593811, -472.7274780, 588.3479004, -795.4498901, 742.7284546
4: -184.5051727, 282.8896484, -405.3910217, 594.0910645, -775.6605835, 679.6483154

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9054748, upper bound: 398.8151986
time: 0.77 seconds

## Relational analysis of NS_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_B1_B1

### Relational analysis result of NS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9137791, upper bound: 398.8637842
time: 1.10 seconds

## Relational analysis of NS_B2_A1_B1_B2

### Relational analysis result of NS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9152031, upper bound: 398.8652542
time: 0.83 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -164.8828735, 273.6366882, -380.5660400, 573.9440308, -734.3572388, 641.4078979
1: -181.5148926, 242.5971832, -416.0835876, 524.8048096, -701.1510620, 647.0748291
2: -181.4661713, 246.9006042, -414.9791260, 533.9889526, -710.2315063, 652.4486694
3: -214.6846161, 279.4453430, -485.5814514, 605.6345215, -814.2229614, 756.3324585
4: -184.8080750, 283.3780212, -416.4398804, 611.7735596, -794.4026489, 691.5787354

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9055065, upper bound: 398.8351220
time: 1.20 seconds

## Relational analysis of NS_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9139819, upper bound: 398.8810653
time: 0.80 seconds

## Relational analysis of NS_B2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9168129, upper bound: 398.8835037
time: 1.25 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -371.6011658, 557.5308228, -381.9024353, 576.5236816, -925.9342651, 916.5483398
1: -405.9487610, 510.1193542, -417.5844421, 526.9909668, -911.2872925, 905.4273071
2: -404.9323730, 519.0952759, -416.4736633, 536.2327881, -921.0347290, 914.8226318
3: -473.1334534, 588.8765259, -487.4274597, 608.1396484, -1061.6729736, 1055.8477783
4: -405.7301636, 594.6139526, -417.9939575, 614.2982788, -1004.9372559, 997.1842041

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8175767, upper bound: 398.8215239
time: 1.12 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8634248, upper bound: 398.8810226
time: 0.94 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -380.8507996, 574.4285889, -382.1474609, 576.9548950, -935.9540405, 934.8452759
1: -416.4064026, 525.2488403, -417.8582153, 527.3723145, -922.3524780, 921.7328491
2: -415.2940674, 534.4339600, -416.7462769, 536.6198730, -932.1108398, 931.5006104
3: -485.9724121, 606.1524658, -487.7610474, 608.5759277, -1075.1625977, 1074.5676270
4: -416.7651062, 612.2853394, -418.2799072, 614.7428589, -1016.7861938, 1015.8733521

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A2_A1

### Relational analysis result of NS_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8217361, upper bound: 398.8807989
time: 1.03 seconds

## Relational analysis of NS_B2_A2_A2_A2

### Relational analysis result of NS_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8810400, upper bound: 398.8811183
time: 0.99 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.69 seconds
NS_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -398.8817404, upper bound: 398.9198913
NS_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -398.9007163, upper bound: 398.9281676
NS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -398.9200177, upper bound: 398.9034343
NS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -398.9285787, upper bound: 398.9285787
NS_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -398.8637842, upper bound: 398.9137791
NS_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -398.8652542, upper bound: 398.9152031
NS_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -398.8810653, upper bound: 398.9139819
NS_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -398.8835037, upper bound: 398.9168129
NS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -398.9137791, upper bound: 398.8637842
NS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -398.9152031, upper bound: 398.8652542
NS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -398.9139819, upper bound: 398.8810653
NS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -398.9168129, upper bound: 398.8835037
NS_B2_A2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.69
Output dim: 0, lower bound: -398.8175767, upper bound: 398.8215239
NS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -398.8634248, upper bound: 398.8810226
NS_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -398.8217361, upper bound: 398.8807989
NS_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.69
Output dim: 0, lower bound: -398.8810400, upper bound: 398.8811183

## BFS NS instance: NS_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -158.7276611, 256.1733704, -161.1894073, 267.6727600, -426.4003906, 417.3627930
1: -174.2021332, 227.3370819, -177.4261169, 237.2196960, -411.4217834, 404.7631836
2: -174.0921478, 231.2686768, -177.3783264, 241.3938751, -415.4860229, 408.6469116
3: -204.3563385, 261.9820862, -209.7526245, 273.2564392, -477.6127625, 471.7346802
4: -176.1709290, 266.0678406, -180.6543884, 277.0518494, -453.2227783, 446.7222290

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817126, upper bound: 398.9032491
time: 0.96 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817126, upper bound: 398.9198913
time: 0.89 seconds

## BFS NS instance: NS_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -151.6383972, 248.3205566, -164.6252747, 273.1701050, -424.8084412, 412.9458313
1: -166.6858826, 220.2291870, -181.2254486, 242.1801758, -408.8660583, 401.4546509
2: -166.6465302, 223.9093781, -181.1805267, 246.4721680, -413.1187134, 405.0899048
3: -196.3782043, 253.6978760, -214.3279419, 278.9593811, -475.3375854, 468.0257568
4: -169.3101044, 257.3973999, -184.5051727, 282.8896484, -452.1997681, 441.9025879

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8946137, upper bound: 398.9032491
time: 0.99 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8946137, upper bound: 398.9281676
time: 0.88 seconds

## BFS NS instance: NS_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -159.8923492, 265.3018188, -169.4530640, 276.8140869, -436.7064209, 434.7548828
1: -175.9862366, 235.1440277, -186.2131653, 245.6297760, -421.6160278, 421.3571777
2: -175.9517822, 239.3302155, -186.0870819, 250.0730896, -426.0247803, 425.4172363
3: -208.0453949, 270.9082947, -219.2395325, 283.0472107, -491.0925903, 490.1478271
4: -179.1729126, 274.7250671, -188.7692108, 287.2324219, -466.4053345, 463.4942627

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_A2_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9034341, upper bound: 398.9034341
time: 0.98 seconds

## Relational analysis of NS_B1_A1_A2_B1_A2

### Relational analysis result of NS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9034341, upper bound: 398.9034343
time: 0.89 seconds

## BFS NS instance: NS_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -163.3380432, 270.8208618, -162.7719269, 269.7117004, -433.0497437, 433.5927734
1: -179.7960052, 240.1278992, -179.1849213, 239.1481323, -418.9440918, 419.3128052
2: -179.7642059, 244.4613647, -179.1102448, 243.4538879, -423.2180786, 423.5715637
3: -212.6234283, 276.6451721, -211.9000854, 275.4799500, -488.1033630, 488.5452576
4: -183.0366974, 280.5787659, -182.3677216, 279.4620361, -462.4986877, 462.9464111

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_A2_B2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9034343, upper bound: 398.9200177
time: 1.07 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9034343, upper bound: 398.9285788
time: 1.12 seconds

## BFS NS instance: NS_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -370.7067261, 553.1517944, -161.1894073, 267.6727600, -624.5137329, 708.4821777
1: -404.7003479, 505.9662476, -177.4261169, 237.2196960, -629.0619507, 676.9814453
2: -403.8412170, 514.9203491, -177.3783264, 241.3938751, -634.2573242, 685.7154541
3: -471.0077515, 584.1878052, -209.7526245, 273.2564392, -734.5821533, 786.3261719
4: -403.8717651, 590.2994995, -180.6543884, 277.0518494, -671.6152954, 767.6813354

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_A1_A1_B1

### Relational analysis result of NS_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8555763, upper bound: 398.9089176
time: 0.87 seconds

## Relational analysis of NS_B1_A2_A1_A1_B2

### Relational analysis result of NS_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8553685, upper bound: 398.8868631
time: 1.03 seconds

## BFS NS instance: NS_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -369.7440796, 553.9658203, -164.6252747, 273.1701050, -629.5552979, 712.8191528
1: -403.8634644, 507.1598816, -181.2254486, 242.1801758, -634.0101318, 682.1756592
2: -402.8531494, 516.1032104, -181.1805267, 246.4721680, -639.3601074, 690.8869629
3: -470.6022034, 585.4978027, -214.3279419, 278.9593811, -740.4638062, 792.4655151
4: -403.5679321, 591.1798096, -184.5051727, 282.8896484, -677.6836548, 772.6427612

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_A1_A2_B1

### Relational analysis result of NS_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8650327, upper bound: 398.9125931
time: 0.76 seconds

## Relational analysis of NS_B1_A2_A1_A2_B2

### Relational analysis result of NS_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8649422, upper bound: 398.8897859
time: 0.76 seconds

## BFS NS instance: NS_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -379.9278870, 569.9880981, -161.4462585, 268.1387939, -634.5479126, 726.7475586
1: -415.1165771, 520.9279785, -177.7146454, 237.6364441, -640.1343994, 693.1074829
2: -414.1694031, 530.0695801, -177.6630707, 241.8217010, -645.3508301, 702.1499023
3: -483.7643433, 601.2854004, -210.1084442, 273.7346497, -748.1264038, 804.8590088
4: -414.8511047, 607.7980347, -180.9562988, 277.5392456, -683.4960327, 786.2311401

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_A2_A1_B1

### Relational analysis result of NS_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8675079, upper bound: 398.9086859
time: 0.95 seconds

## Relational analysis of NS_B1_A2_A2_A1_B2

### Relational analysis result of NS_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8673140, upper bound: 398.8870123
time: 0.93 seconds

## BFS NS instance: NS_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -378.9644165, 570.8218384, -164.8828735, 273.6366882, -639.6145020, 731.0882568
1: -414.2875061, 522.2532959, -181.5148926, 242.5971832, -645.1090698, 698.4819336
2: -413.1824036, 531.4051514, -181.4661713, 246.9006042, -650.4798584, 707.5289917
3: -483.4048462, 602.7315063, -214.6846161, 279.4453430, -754.0358276, 811.1809082
4: -414.5777283, 608.8121338, -184.8080750, 283.3780212, -689.5748291, 791.3339233

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_A2_A2_B1

### Relational analysis result of NS_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8821928, upper bound: 398.9141074
time: 1.06 seconds

## Relational analysis of NS_B1_A2_A2_A2_B2

### Relational analysis result of NS_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8818822, upper bound: 398.8907469
time: 0.93 seconds

## BFS NS instance: NS_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -161.1894073, 267.6727600, -370.7067261, 553.1517944, -708.4821777, 624.5137329
1: -177.4261169, 237.2196960, -404.7003479, 505.9662476, -676.9813843, 629.0619507
2: -177.3783264, 241.3938751, -403.8412170, 514.9203491, -685.7155151, 634.2573242
3: -209.7526245, 273.2564392, -471.0077515, 584.1878052, -786.3261719, 734.5821533
4: -180.6543884, 277.0518494, -403.8717651, 590.2994995, -767.6813354, 671.6152344

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_B1_B1_A1

### Relational analysis result of NS_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9089176, upper bound: 398.8555763
time: 0.93 seconds

## Relational analysis of NS_B2_A1_B1_B1_A2

### Relational analysis result of NS_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8868631, upper bound: 398.8553685
time: 0.92 seconds

## BFS NS instance: NS_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -164.6252747, 273.1701050, -369.7440796, 553.9658203, -712.8190918, 629.5552979
1: -181.2254486, 242.1801758, -403.8634644, 507.1598816, -682.1756592, 634.0100708
2: -181.1805267, 246.4721680, -402.8531494, 516.1032104, -690.8869019, 639.3601685
3: -214.3279419, 278.9593811, -470.6022034, 585.4978027, -792.4655151, 740.4638062
4: -184.5051727, 282.8896484, -403.5679321, 591.1798096, -772.6427612, 677.6836548

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_B1_B2_A1

### Relational analysis result of NS_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9125931, upper bound: 398.8650327
time: 0.95 seconds

## Relational analysis of NS_B2_A1_B1_B2_A2

### Relational analysis result of NS_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8897859, upper bound: 398.8649422
time: 0.81 seconds

## BFS NS instance: NS_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -161.4462585, 268.1387939, -379.9278870, 569.9880981, -726.7475586, 634.5479126
1: -177.7146454, 237.6364441, -415.1165771, 520.9279785, -693.1074219, 640.1343994
2: -177.6630707, 241.8217010, -414.1694031, 530.0695801, -702.1499023, 645.3508301
3: -210.1084442, 273.7346497, -483.7643433, 601.2854004, -804.8590088, 748.1263428
4: -180.9562988, 277.5392456, -414.8511047, 607.7980347, -786.2311401, 683.4960327

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_B2_B1_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9086859, upper bound: 398.8675079
time: 0.78 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2

### Relational analysis result of NS_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8870123, upper bound: 398.8673140
time: 0.79 seconds

## BFS NS instance: NS_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -164.8828735, 273.6366882, -378.9644165, 570.8218384, -731.0882568, 639.6145020
1: -181.5148926, 242.5971832, -414.2875061, 522.2532959, -698.4819946, 645.1091309
2: -181.4661713, 246.9006042, -413.1824036, 531.4051514, -707.5289307, 650.4797974
3: -214.6846161, 279.4453430, -483.4048462, 602.7315063, -811.1809082, 754.0358887
4: -184.8080750, 283.3780212, -414.5777283, 608.8121338, -791.3339233, 689.5748291

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_B2_B2_A1

### Relational analysis result of NS_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9141074, upper bound: 398.8821928
time: 0.89 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2

### Relational analysis result of NS_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8907469, upper bound: 398.8818822
time: 0.96 seconds

## BFS NS instance: NS_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -371.6011658, 557.5308228, -381.7839661, 576.3209839, -925.7205811, 916.4107666
1: -405.9487610, 510.1193542, -417.4550171, 526.8139038, -911.0989990, 905.2796021
2: -404.9323730, 519.0952759, -416.3414307, 536.0538940, -920.8440552, 914.6740723
3: -473.1334534, 588.8765259, -487.2724609, 607.9384766, -1061.4587402, 1055.6763916
4: -405.7301636, 594.6139526, -417.8598022, 614.0927734, -1004.7222290, 997.0383301

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A1_B2_A1

### Relational analysis result of NS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8153689, upper bound: 398.8807347
time: 1.00 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2

### Relational analysis result of NS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8153689, upper bound: 398.8810226
time: 1.12 seconds

## BFS NS instance: NS_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -366.5205078, 551.9121094, -379.8778992, 573.5802002, -918.0713501, 909.7348633
1: -400.8809204, 505.0452881, -415.3908081, 524.2570801, -903.4597778, 898.7125244
2: -399.7124939, 513.6216431, -414.2825012, 533.4310913, -913.1044312, 907.9106445
3: -468.0121765, 582.7849731, -484.9009094, 604.9683838, -1053.4075928, 1048.2987061
4: -401.1957397, 588.9085693, -415.8319397, 611.1199951, -997.7414551, 989.7777100

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8208053, upper bound: 398.8215892
time: 1.07 seconds

## Relational analysis of NS_B2_A2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8208053, upper bound: 398.8807989
time: 0.87 seconds

## BFS NS instance: NS_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -380.7317200, 574.2249146, -382.1474609, 576.9548950, -935.8154907, 934.6303101
1: -416.2763977, 525.0709229, -417.8582153, 527.3723145, -922.2035522, 921.5433350
2: -415.1612549, 534.2542725, -416.7462769, 536.6198730, -931.9613037, 931.3090820
3: -485.8170166, 605.9503174, -487.7610474, 608.5759277, -1074.9903564, 1074.3519287
4: -416.6303406, 612.0791016, -418.2799072, 614.7428589, -1016.6395264, 1015.6572266

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_B2_A2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -398.8268403, upper bound: 398.8216305
time: 1.20 seconds

## Relational analysis of NS_B2_A2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8268403, upper bound: 398.8811183
time: 0.94 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.84 seconds
NS_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8817126, upper bound: 398.9032491
NS_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8817126, upper bound: 398.9198913
NS_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8946137, upper bound: 398.9032491
NS_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8946137, upper bound: 398.9281676
NS_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.9034341, upper bound: 398.9034341
NS_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.9034341, upper bound: 398.9034343
NS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.9034343, upper bound: 398.9200177
NS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.9034343, upper bound: 398.9285788
NS_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8555763, upper bound: 398.9089176
NS_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8553685, upper bound: 398.8868631
NS_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8650327, upper bound: 398.9125931
NS_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8649422, upper bound: 398.8897859
NS_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8675079, upper bound: 398.9086859
NS_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8673140, upper bound: 398.8870123
NS_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8821928, upper bound: 398.9141074
NS_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8818822, upper bound: 398.8907469
NS_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.9089176, upper bound: 398.8555763
NS_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8868631, upper bound: 398.8553685
NS_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.9125931, upper bound: 398.8650327
NS_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8897859, upper bound: 398.8649422
NS_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.9086859, upper bound: 398.8675079
NS_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8870123, upper bound: 398.8673140
NS_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.9141074, upper bound: 398.8821928
NS_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8907469, upper bound: 398.8818822
NS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8153689, upper bound: 398.8807347
NS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8153689, upper bound: 398.8810226
NS_B2_A2_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8208053, upper bound: 398.8215892
NS_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8208053, upper bound: 398.8807989
NS_B2_A2_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8268403, upper bound: 398.8216305
NS_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.84
Output dim: 0, lower bound: -398.8268403, upper bound: 398.8811183

## BFS NS instance: NS_B1_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -158.7276611, 256.1733704, -169.1950378, 276.3427734, -435.0704346, 425.3684082
1: -174.2021332, 227.3370819, -185.9232788, 245.2077942, -419.4099121, 413.2603760
2: -174.0921478, 231.2686768, -185.8009796, 249.6407318, -423.7328796, 417.0696411
3: -204.3563385, 261.9820862, -218.8775940, 282.5628662, -486.9191895, 480.8596802
4: -176.1709290, 266.0678406, -188.4655304, 286.7400818, -462.9109802, 454.5333557

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A1_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793442, upper bound: 398.8903971
time: 0.80 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8748987, upper bound: 398.8912387
time: 0.97 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -158.7276611, 256.1733704, -162.5143433, 269.2449036, -427.9725647, 418.6877136
1: -174.2021332, 227.3370819, -178.8953094, 238.7306671, -412.9328003, 406.2323914
2: -174.0921478, 231.2686768, -178.8245087, 243.0179749, -417.1101074, 410.0931702
3: -204.3563385, 261.9820862, -211.5433044, 274.9932861, -479.3496094, 473.5253906
4: -176.1709290, 266.0678406, -182.0640717, 278.9735107, -455.1444397, 448.1318970

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A1_A1_B2_A1

### Relational analysis result of NS_B1_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8793442, upper bound: 398.8903971
time: 0.83 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_A2

### Relational analysis result of NS_B1_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8748987, upper bound: 398.8912387
time: 0.92 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -151.6383972, 248.3205566, -169.1950378, 276.3427734, -427.9811401, 417.5155945
1: -166.6858826, 220.2291870, -185.9232788, 245.2077942, -411.8936768, 406.1524353
2: -166.6465302, 223.9093781, -185.8009796, 249.6407318, -416.2872314, 409.7103271
3: -196.3782043, 253.6978760, -218.8775940, 282.5628662, -478.9410400, 472.5753784
4: -169.3101044, 257.3973999, -188.4655304, 286.7400818, -456.0501709, 445.8629150

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A1_A2_B1_B1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8653017, upper bound: 398.8915267
time: 1.43 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A1_A2_B1_B1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8742275, upper bound: 398.8999922
time: 0.76 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8749529, upper bound: 398.8912387
time: 0.87 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -151.6383972, 248.3205566, -162.5143433, 269.2449036, -420.8832703, 410.8348999
1: -166.6858826, 220.2291870, -178.8953094, 238.7306671, -405.4165649, 399.1244812
2: -166.6465302, 223.9093781, -178.8245087, 243.0179749, -409.6644897, 402.7338867
3: -196.3782043, 253.6978760, -211.5433044, 274.9932861, -471.3714905, 465.2411194
4: -169.3101044, 257.3973999, -182.0640717, 278.9735107, -448.2836304, 439.4614563

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A1_A2_B2_B1

### Relational analysis result of NS_B1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8653017, upper bound: 398.9103379
time: 0.79 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A1_A2_B2_A1

### Relational analysis result of NS_B1_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8929836, upper bound: 398.8911555
time: 1.18 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2_A2

### Relational analysis result of NS_B1_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8749529, upper bound: 398.8912387
time: 1.04 seconds

## BFS NS instance: NS_B1_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -167.9532776, 274.1008301, -169.4530640, 276.8140869, -444.7673340, 443.5538940
1: -184.5478210, 243.2079163, -186.2131653, 245.6297760, -430.1776123, 429.4210815
2: -184.4354706, 247.6372070, -186.0870819, 250.0730896, -434.5084839, 433.7243042
3: -217.2633972, 280.2965088, -219.2395325, 283.0472107, -500.3106079, 499.5360413
4: -187.0465240, 284.5088806, -188.7692108, 287.2324219, -474.2789307, 473.2780762

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A2_B1_A1_B1

### Relational analysis result of NS_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8905469, upper bound: 398.8999875
time: 0.78 seconds

## Relational analysis of NS_B1_A1_A2_B1_A1_B2

### Relational analysis result of NS_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8913861, upper bound: 398.8917516
time: 0.84 seconds

## BFS NS instance: NS_B1_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -161.2266846, 266.9071960, -169.4530640, 276.8140869, -438.0407715, 436.3602600
1: -177.4663391, 236.6997681, -186.2131653, 245.6297760, -423.0960999, 422.9129333
2: -177.4091339, 241.0200500, -186.0870819, 250.0730896, -427.4822388, 427.1071167
3: -209.8498840, 272.6787109, -219.2395325, 283.0472107, -492.8970947, 491.9182129
4: -180.5965576, 276.6629028, -188.7692108, 287.2324219, -467.8289795, 465.4321289

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A2_B1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8905469, upper bound: 398.8999875
time: 0.73 seconds

## Relational analysis of NS_B1_A1_A2_B1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8913861, upper bound: 398.8917516
time: 0.86 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -167.9532776, 274.1008301, -162.7719269, 269.7117004, -437.6649475, 436.8727417
1: -184.5478210, 243.2079163, -179.1849213, 239.1481323, -423.6959534, 422.3928223
2: -184.4354706, 247.6372070, -179.1102448, 243.4538879, -427.8892822, 426.7474365
3: -217.2633972, 280.2965088, -211.9000854, 275.4799500, -492.7433167, 492.1965027
4: -187.0465240, 284.5088806, -182.3677216, 279.4620361, -466.5085144, 466.8765259

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A2_B2_A1_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8999875, upper bound: 398.8905470
time: 1.04 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_A2

### Relational analysis result of NS_B1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8913861, upper bound: 398.8913861
time: 1.11 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -161.2266846, 266.9071960, -162.7719269, 269.7117004, -430.9383545, 429.6791382
1: -177.4663391, 236.6997681, -179.1849213, 239.1481323, -416.6144409, 415.8847046
2: -177.4091339, 241.0200500, -179.1102448, 243.4538879, -420.8630371, 420.1302490
3: -209.8498840, 272.6787109, -211.9000854, 275.4799500, -485.3298340, 484.5786743
4: -180.5965576, 276.6629028, -182.3677216, 279.4620361, -460.0585938, 459.0305786

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8905469, upper bound: 398.9000666
time: 0.98 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8913861, upper bound: 398.8913951
time: 0.91 seconds

## BFS NS instance: NS_B1_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -370.4505310, 552.8575439, -157.2507782, 260.0291748, -616.5748901, 704.3623657
1: -404.4254150, 505.6880798, -173.0497284, 230.6341248, -622.2855225, 672.4061890
2: -403.5682373, 514.6397705, -172.9300690, 234.8193207, -627.4832764, 681.0685425
3: -470.7047729, 583.8631592, -204.4637909, 265.7083740, -726.6928101, 780.7687988
4: -403.6124878, 589.9713135, -176.0269165, 269.4380188, -663.7185059, 762.7736816

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A1_A1_B1_A1

### Relational analysis result of NS_B1_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8334644, upper bound: 398.8958155
time: 1.15 seconds

## Relational analysis of NS_B1_A2_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_A1_A1_B1_B1

### Relational analysis result of NS_B1_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8548668, upper bound: 398.9086190
time: 1.01 seconds

## Relational analysis of NS_B1_A2_A1_A1_B1_B2

### Relational analysis result of NS_B1_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8548234, upper bound: 398.8895768
time: 1.36 seconds

## BFS NS instance: NS_B1_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -369.3327026, 550.1704712, -178.7428131, 288.6892090, -643.6879883, 722.5952759
1: -403.1189575, 503.4743042, -196.1391296, 256.6153870, -647.2892456, 692.8065186
2: -402.2566528, 512.4144897, -196.2842865, 261.4315796, -653.0009766, 701.8288574
3: -469.0070801, 581.3478394, -230.7044983, 295.7806702, -755.4017944, 804.1354980
4: -402.1409912, 587.3819580, -198.5739288, 300.5153809, -693.2806396, 782.3081665

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_A1_A1_B2_B1

### Relational analysis result of NS_B1_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8384215, upper bound: 398.8640084
time: 1.06 seconds

## Relational analysis of NS_B1_A2_A1_A1_B2_B2

### Relational analysis result of NS_B1_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8504049, upper bound: 398.8868631
time: 0.89 seconds

## BFS NS instance: NS_B1_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -369.6340332, 553.8275757, -160.6211395, 265.4533081, -621.6854248, 708.7713013
1: -403.7419434, 507.0274658, -176.7797394, 235.5208893, -627.2984619, 677.6622925
2: -402.7339172, 515.9719238, -176.6647949, 239.8218536, -632.6459961, 686.3027954
3: -470.4639587, 585.3422241, -208.9590149, 271.3406067, -732.6458740, 786.9856567
4: -403.4518127, 591.0261230, -179.8074036, 275.1850891, -669.8370972, 767.8262939

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_A1_A2_B1_A1

### Relational analysis result of NS_B1_A2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8635669, upper bound: 398.9117022
time: 1.09 seconds

## Relational analysis of NS_B1_A2_A1_A2_B1_A2

### Relational analysis result of NS_B1_A2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8650327, upper bound: 398.9125931
time: 0.98 seconds

## BFS NS instance: NS_B1_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -367.2177734, 549.5583496, -182.2581024, 294.1833801, -647.6856079, 725.5526733
1: -401.0744019, 503.2326355, -200.0035400, 261.6096802, -651.1149292, 696.5737305
2: -400.0389404, 512.1475220, -200.1821289, 266.4823303, -656.8988647, 705.5755005
3: -467.2430115, 580.9985352, -235.2860718, 301.5429382, -760.0814819, 808.5729370
4: -400.6291504, 586.6339111, -202.5496979, 306.3214417, -698.1830444, 785.7445068

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_A1_A2_B2_A1

### Relational analysis result of NS_B1_A2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8635097, upper bound: 398.8875381
time: 0.78 seconds

## Relational analysis of NS_B1_A2_A1_A2_B2_A2

### Relational analysis result of NS_B1_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8649422, upper bound: 398.8875607
time: 1.93 seconds

## BFS NS instance: NS_B1_A2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -379.6812439, 569.7057495, -157.5080261, 260.4975586, -626.6222534, 722.6414185
1: -414.8514099, 520.6612549, -173.3386383, 231.0528259, -633.3713989, 688.5454712
2: -413.9067383, 529.8004150, -173.2155914, 235.2493286, -638.5895996, 697.5159302
3: -483.4714661, 600.9736328, -204.8202667, 266.1954651, -740.2511597, 799.3165283
4: -414.6011353, 607.4816284, -176.3299103, 269.9279175, -675.6112061, 781.3374023

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_A2_A1_B1_B1

### Relational analysis result of NS_B1_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8672898, upper bound: 398.9083145
time: 0.98 seconds

## Relational analysis of NS_B1_A2_A2_A1_B1_B2

### Relational analysis result of NS_B1_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8672419, upper bound: 398.8896117
time: 1.14 seconds

## BFS NS instance: NS_B1_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -378.5715332, 567.0715332, -179.0177155, 289.1788330, -653.7701416, 740.9316406
1: -413.5740356, 518.5043945, -196.4476929, 257.0481262, -658.4136353, 709.0304565
2: -412.6114807, 527.6301270, -196.5879669, 261.8765259, -664.1392212, 718.3666382
3: -481.8307495, 598.5318604, -231.0807953, 296.2780151, -769.0272827, 822.7891235
4: -413.1810303, 604.9837646, -198.8889160, 301.0144043, -705.2481689, 800.9642944

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_A2_A1_B2_B1

### Relational analysis result of NS_B1_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8600052, upper bound: 398.8690146
time: 1.17 seconds

## Relational analysis of NS_B1_A2_A2_A1_B2_B2

### Relational analysis result of NS_B1_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8645381, upper bound: 398.8870123
time: 0.84 seconds

## BFS NS instance: NS_B1_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -378.8833313, 570.7182007, -160.8784637, 265.9215088, -631.7738037, 727.0757446
1: -414.1966248, 522.1527710, -177.0686951, 235.9393463, -638.4287720, 694.0011597
2: -413.0941467, 531.3062134, -176.9501801, 240.2561340, -643.7967529, 702.9758911
3: -483.3000793, 602.6131592, -209.3154907, 271.8285217, -746.2537842, 805.7390747
4: -414.4901123, 608.6961670, -180.1105499, 275.6749573, -681.7576904, 786.5562744

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_A2_A2_B1_A1

### Relational analysis result of NS_B1_A2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8811989, upper bound: 398.9137580
time: 0.89 seconds

## Relational analysis of NS_B1_A2_A2_A2_B1_A2

### Relational analysis result of NS_B1_A2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8821928, upper bound: 398.9141074
time: 0.74 seconds

## BFS NS instance: NS_B1_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -376.5573730, 566.5901489, -182.5338745, 294.6768799, -657.8974609, 743.9966431
1: -411.6289673, 518.5187378, -200.3133240, 262.0391235, -662.3648071, 713.0778198
2: -410.4987488, 527.6481323, -200.4873199, 266.9307861, -668.1702271, 722.4322510
3: -480.2236633, 598.4577026, -235.6646729, 302.0430603, -773.8438110, 827.5274048
4: -411.8022461, 604.5101929, -202.8659515, 306.8235168, -710.2513428, 804.6715088

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_A2_A2_B2_A1

### Relational analysis result of NS_B1_A2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8810380, upper bound: 398.8883656
time: 0.80 seconds

## Relational analysis of NS_B1_A2_A2_A2_B2_A2

### Relational analysis result of NS_B1_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8818822, upper bound: 398.8883585
time: 1.04 seconds

## BFS NS instance: NS_B2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -157.2507782, 260.0291748, -370.4505310, 552.8575439, -704.3623657, 616.5749512
1: -173.0497284, 230.6341248, -404.4254150, 505.6880798, -672.4061279, 622.2855225
2: -172.9300690, 234.8193207, -403.5682373, 514.6397705, -681.0685425, 627.4832764
3: -204.4637909, 265.7083740, -470.7047729, 583.8631592, -780.7687988, 726.6927490
4: -176.0269165, 269.4380188, -403.6124878, 589.9713135, -762.7736816, 663.7185059

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B1_B1_A1_B1

### Relational analysis result of NS_B2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8958155, upper bound: 398.8334644
time: 0.83 seconds

## Relational analysis of NS_B2_A1_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_B1_B1_A1_A1

### Relational analysis result of NS_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9086190, upper bound: 398.8548668
time: 1.11 seconds

## Relational analysis of NS_B2_A1_B1_B1_A1_A2

### Relational analysis result of NS_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8895768, upper bound: 398.8548234
time: 0.95 seconds

## BFS NS instance: NS_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -178.7428131, 288.6892090, -369.3327026, 550.1704712, -722.5953369, 643.6879883
1: -196.1391296, 256.6153870, -403.1189575, 503.4743042, -692.8065186, 647.2891846
2: -196.2842865, 261.4315796, -402.2566528, 512.4144897, -701.8288574, 653.0008545
3: -230.7044983, 295.7806702, -469.0070801, 581.3478394, -804.1354980, 755.4017944
4: -198.5739288, 300.5153809, -402.1409912, 587.3819580, -782.3081665, 693.2806396

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B1_B1_A2_A1

### Relational analysis result of NS_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8640084, upper bound: 398.8384215
time: 1.13 seconds

## Relational analysis of NS_B2_A1_B1_B1_A2_A2

### Relational analysis result of NS_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8868631, upper bound: 398.8504049
time: 0.98 seconds

## BFS NS instance: NS_B2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -160.6211395, 265.4533081, -369.6340332, 553.8275757, -708.7713623, 621.6853638
1: -176.7797394, 235.5208893, -403.7419434, 507.0274658, -677.6622925, 627.2985229
2: -176.6647949, 239.8218536, -402.7339172, 515.9719238, -686.3027954, 632.6459351
3: -208.9590149, 271.3406067, -470.4639587, 585.3422241, -786.9857178, 732.6458740
4: -179.8074036, 275.1850891, -403.4518127, 591.0261230, -767.8262939, 669.8370972

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B1_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9117022, upper bound: 398.8635669
time: 0.98 seconds

## Relational analysis of NS_B2_A1_B1_B2_A1_B2

### Relational analysis result of NS_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9125931, upper bound: 398.8650327
time: 1.32 seconds

## BFS NS instance: NS_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -182.2581024, 294.1833801, -367.2177734, 549.5583496, -725.5526733, 647.6856079
1: -200.0035400, 261.6096802, -401.0744019, 503.2326355, -696.5737915, 651.1149902
2: -200.1821289, 266.4823303, -400.0389404, 512.1475220, -705.5754395, 656.8988037
3: -235.2860718, 301.5429382, -467.2430115, 580.9985352, -808.5729980, 760.0814819
4: -202.5496979, 306.3214417, -400.6291504, 586.6339111, -785.7445679, 698.1830444

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B1_B2_A2_B1

### Relational analysis result of NS_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8875381, upper bound: 398.8635097
time: 0.79 seconds

## Relational analysis of NS_B2_A1_B1_B2_A2_B2

### Relational analysis result of NS_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8875607, upper bound: 398.8649422
time: 1.01 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -157.5080261, 260.4975586, -379.6812439, 569.7057495, -722.6413574, 626.6222534
1: -173.3386383, 231.0528259, -414.8514099, 520.6612549, -688.5454712, 633.3713989
2: -173.2155914, 235.2493286, -413.9067383, 529.8004150, -697.5159302, 638.5895996
3: -204.8202667, 266.1954651, -483.4714661, 600.9736328, -799.3165283, 740.2512207
4: -176.3299103, 269.9279175, -414.6011353, 607.4816284, -781.3374023, 675.6112061

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_B2_B1_A1_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9083145, upper bound: 398.8672898
time: 0.86 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8896117, upper bound: 398.8672419
time: 1.03 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -179.0177155, 289.1788330, -378.5715332, 567.0715332, -740.9315796, 653.7701416
1: -196.4476929, 257.0481262, -413.5740356, 518.5043945, -709.0304565, 658.4136353
2: -196.5879669, 261.8765259, -412.6114807, 527.6301270, -718.3666382, 664.1392822
3: -231.0807953, 296.2780151, -481.8307495, 598.5318604, -822.7890625, 769.0272827
4: -198.8889160, 301.0144043, -413.1810303, 604.9837646, -800.9642944, 705.2481689

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B2_B1_A2_A1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8690146, upper bound: 398.8600052
time: 0.97 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8870123, upper bound: 398.8645381
time: 1.07 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -160.8784637, 265.9215088, -378.8833313, 570.7182007, -727.0757446, 631.7738647
1: -177.0686951, 235.9393463, -414.1966553, 522.1527710, -694.0011597, 638.4287720
2: -176.9501801, 240.2561340, -413.0941772, 531.3062744, -702.9758911, 643.7967529
3: -209.3154907, 271.8285217, -483.3001404, 602.6130981, -805.7390137, 746.2537842
4: -180.1105499, 275.6749573, -414.4900818, 608.6961670, -786.5562744, 681.7577515

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B2_B2_A1_B1

### Relational analysis result of NS_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9137580, upper bound: 398.8811989
time: 0.94 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9141074, upper bound: 398.8821928
time: 0.98 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -182.5338745, 294.6768799, -376.5573730, 566.5901489, -743.9966431, 657.8975220
1: -200.3133240, 262.0391235, -411.6289673, 518.5187378, -713.0778198, 662.3648071
2: -200.4873199, 266.9307861, -410.4987488, 527.6481323, -722.4322510, 668.1702271
3: -235.6646729, 302.0430603, -480.2236633, 598.4577026, -827.5274048, 773.8437500
4: -202.8659515, 306.8235168, -411.8022461, 604.5101929, -804.6715698, 710.2513428

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B2_B2_A2_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8883656, upper bound: 398.8810380
time: 0.99 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8883585, upper bound: 398.8818822
time: 1.17 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -357.6149292, 535.5872803, -381.7829895, 576.3195801, -911.4532471, 893.9543457
1: -390.8083191, 490.4670715, -417.4539795, 526.8126221, -895.6436157, 885.1333618
2: -389.7409058, 498.8159485, -416.3404846, 536.0525513, -905.2852783, 893.9449463
3: -455.6061401, 566.1340942, -487.2713013, 607.9371948, -1043.6086426, 1032.7952881
4: -390.5436096, 571.7918701, -417.8588257, 614.0912476, -989.6569214, 973.8140259

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A1_B2_A1_A1

### Relational analysis result of NS_B2_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8131470, upper bound: 398.8805967
time: 0.83 seconds

## Relational analysis of NS_B2_A2_A1_B2_A1_A2

### Relational analysis result of NS_B2_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8131474, upper bound: 398.8807346
time: 1.00 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -371.4789429, 557.3206177, -381.7839661, 576.3209839, -925.5792236, 916.1898193
1: -405.8154297, 509.9360657, -417.4550171, 526.8139038, -910.9468384, 905.0853271
2: -404.7963562, 518.9102783, -416.3414307, 536.0538940, -920.6912231, 914.4776001
3: -472.9738464, 588.6682739, -487.2724609, 607.9384766, -1061.2847900, 1055.4553223
4: -405.5914612, 594.4014282, -417.8598022, 614.0927734, -1004.5723267, 996.8163452

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A1_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8131470, upper bound: 398.8793485
time: 0.81 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2_A2

### Relational analysis result of NS_B2_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8131476, upper bound: 398.8808937
time: 0.95 seconds

## BFS NS instance: NS_B2_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -366.5205078, 551.9121094, -382.0289001, 576.7522583, -921.0632935, 911.6636963
1: -400.8809204, 505.0452881, -417.7287598, 527.1953125, -906.2318726, 900.8481445
2: -399.7124939, 513.6216431, -416.6140137, 536.4409790, -915.9497681, 910.0512085
3: -468.0121765, 582.7849731, -487.6061401, 608.3746338, -1056.6240234, 1050.8460693
4: -401.1957397, 588.9085693, -418.1456299, 614.5377197, -1001.0549316, 991.9233398

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A2_A1_B2_A1

### Relational analysis result of NS_B2_A2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8197001, upper bound: 398.8806620
time: 1.06 seconds

## Relational analysis of NS_B2_A2_A2_A1_B2_A2

### Relational analysis result of NS_B2_A2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8198714, upper bound: 398.8807989
time: 1.00 seconds

## BFS NS instance: NS_B2_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -380.7317200, 574.2249146, -382.0289001, 576.7522583, -935.6019897, 934.4927979
1: -416.2763977, 525.0709229, -417.7287598, 527.1953125, -922.0152588, 921.3953857
2: -415.1612549, 534.2542725, -416.6140137, 536.4409790, -931.7706299, 931.1604004
3: -485.8170166, 605.9503174, -487.6061401, 608.3746338, -1074.7762451, 1074.1807861
4: -416.6303406, 612.0791016, -418.1456299, 614.5377197, -1016.4246216, 1015.5114136

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A2_A2_B2_B1

### Relational analysis result of NS_B2_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8267983, upper bound: 398.8790301
time: 0.86 seconds

## Relational analysis of NS_B2_A2_A2_A2_B2_B2

### Relational analysis result of NS_B2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8205160, upper bound: 398.8810410
time: 1.14 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.74 seconds
NS_B1_A1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8793442, upper bound: 398.8903971
NS_B1_A1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8748987, upper bound: 398.8912387
NS_B1_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8793442, upper bound: 398.8903971
NS_B1_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8748987, upper bound: 398.8912387
NS_B1_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8742275, upper bound: 398.8999922
NS_B1_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8749529, upper bound: 398.8912387
NS_B1_A1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8929836, upper bound: 398.8911555
NS_B1_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8749529, upper bound: 398.8912387
NS_B1_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8905469, upper bound: 398.8999875
NS_B1_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8913861, upper bound: 398.8917516
NS_B1_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8905469, upper bound: 398.8999875
NS_B1_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8913861, upper bound: 398.8917516
NS_B1_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8999875, upper bound: 398.8905470
NS_B1_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8913861, upper bound: 398.8913861
NS_B1_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8905469, upper bound: 398.9000666
NS_B1_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8913861, upper bound: 398.8913951
NS_B1_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8548668, upper bound: 398.9086190
NS_B1_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8548234, upper bound: 398.8895768
NS_B1_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8384215, upper bound: 398.8640084
NS_B1_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8504049, upper bound: 398.8868631
NS_B1_A2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8635669, upper bound: 398.9117022
NS_B1_A2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8650327, upper bound: 398.9125931
NS_B1_A2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8635097, upper bound: 398.8875381
NS_B1_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8649422, upper bound: 398.8875607
NS_B1_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8672898, upper bound: 398.9083145
NS_B1_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8672419, upper bound: 398.8896117
NS_B1_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8600052, upper bound: 398.8690146
NS_B1_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8645381, upper bound: 398.8870123
NS_B1_A2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8811989, upper bound: 398.9137580
NS_B1_A2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8821928, upper bound: 398.9141074
NS_B1_A2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8810380, upper bound: 398.8883656
NS_B1_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8818822, upper bound: 398.8883585
NS_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.9086190, upper bound: 398.8548668
NS_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8895768, upper bound: 398.8548234
NS_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8640084, upper bound: 398.8384215
NS_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8868631, upper bound: 398.8504049
NS_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.9117022, upper bound: 398.8635669
NS_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.9125931, upper bound: 398.8650327
NS_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8875381, upper bound: 398.8635097
NS_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8875607, upper bound: 398.8649422
NS_B2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.9083145, upper bound: 398.8672898
NS_B2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8896117, upper bound: 398.8672419
NS_B2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8690146, upper bound: 398.8600052
NS_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8870123, upper bound: 398.8645381
NS_B2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.9137580, upper bound: 398.8811989
NS_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.9141074, upper bound: 398.8821928
NS_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8883656, upper bound: 398.8810380
NS_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8883585, upper bound: 398.8818822
NS_B2_A2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8131470, upper bound: 398.8805967
NS_B2_A2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8131474, upper bound: 398.8807346
NS_B2_A2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8131470, upper bound: 398.8793485
NS_B2_A2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8131476, upper bound: 398.8808937
NS_B2_A2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8197001, upper bound: 398.8806620
NS_B2_A2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8198714, upper bound: 398.8807989
NS_B2_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8267983, upper bound: 398.8790301
NS_B2_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.74
Output dim: 0, lower bound: -398.8205160, upper bound: 398.8810410

## BFS NS instance: NS_B1_A1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -156.1449127, 251.0198059, -169.1950378, 276.3427734, -432.4876709, 420.2148132
1: -171.3356628, 222.9051056, -185.9232788, 245.2077942, -416.5434570, 408.8283081
2: -171.1851196, 226.8253632, -185.8009796, 249.6407318, -420.8258057, 412.6263428
3: -200.8874664, 256.8955994, -218.8775940, 282.5628662, -483.4502869, 475.7731323
4: -173.1624146, 260.9781189, -188.4655304, 286.7400818, -459.9023743, 449.4436340

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A1_A1_B1_A1_B1

### Relational analysis result of NS_B1_A1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8742491, upper bound: 398.8916180
time: 0.96 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_A1_B2

### Relational analysis result of NS_B1_A1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8742491, upper bound: 398.8916180
time: 0.85 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -158.3645020, 255.9281158, -166.4114990, 272.4227600, -430.7872314, 422.3395996
1: -173.7190857, 227.0350189, -182.8413849, 241.6236420, -415.3426819, 409.8764038
2: -173.6867828, 230.9221497, -182.7624359, 245.9811096, -419.6678467, 413.6845703
3: -203.8833466, 261.6620178, -215.3116913, 278.4302063, -482.3135376, 476.9736938
4: -175.7831421, 265.6315613, -185.4352875, 282.4794617, -458.2626038, 451.0668335

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8742491, upper bound: 398.8916204
time: 0.89 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8742491, upper bound: 398.8916204
time: 0.91 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -156.1449127, 251.0198059, -162.5143433, 269.2449036, -425.3898315, 413.5340576
1: -171.3356628, 222.9051056, -178.8953094, 238.7306671, -410.0662842, 401.8003540
2: -171.1851196, 226.8253632, -178.8245087, 243.0179749, -414.2030945, 405.6498718
3: -200.8874664, 256.8955994, -211.5433044, 274.9932861, -475.8807068, 468.4389038
4: -173.1624146, 260.9781189, -182.0640717, 278.9735107, -452.1358948, 443.0421753

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8750163, upper bound: 398.8903971
time: 0.91 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B2

### Relational analysis result of NS_B1_A1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8750163, upper bound: 398.8903971
time: 0.84 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -158.3645020, 255.9281158, -160.2622986, 266.2265930, -424.5910645, 416.1903992
1: -173.7190857, 227.0350189, -176.4176178, 235.9544373, -409.6735229, 403.4526367
2: -173.6867828, 230.9221497, -176.3856354, 240.1705780, -413.8573608, 407.3078003
3: -203.8833466, 261.6620178, -208.7162018, 271.7940063, -475.6773071, 470.3781738
4: -175.7831421, 265.6315613, -179.6582489, 275.6768188, -451.4598999, 445.2897949

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_A1

### Relational analysis result of NS_B1_A1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8742785, upper bound: 398.8903064
time: 1.04 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B1

### Relational analysis result of NS_B1_A1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8750163, upper bound: 398.8912387
time: 1.27 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B2

### Relational analysis result of NS_B1_A1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8750163, upper bound: 398.8912387
time: 1.06 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -151.6383972, 248.3205566, -166.6065979, 271.2405701, -422.8789062, 414.9271545
1: -166.6858826, 220.2291870, -183.0555725, 240.7987671, -407.4846497, 403.2847290
2: -166.6465302, 223.9093781, -182.8773804, 245.2355652, -411.8820801, 406.7866516
3: -196.3782043, 253.6978760, -215.4282684, 277.4872437, -473.8653870, 469.1261292
4: -169.3101044, 257.3973999, -185.4318542, 281.6902771, -451.0003662, 442.8292542

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8543259, upper bound: 398.8939104
time: 1.22 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8616591, upper bound: 398.8938938
time: 1.02 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -149.5401764, 245.4983368, -168.3358459, 275.4323730, -424.9724731, 413.8341675
1: -164.3741760, 217.6149292, -184.8941803, 244.3740692, -408.7481689, 402.5090942
2: -164.3608856, 221.2577209, -184.8663177, 248.7947693, -413.1556091, 406.1240234
3: -193.7419739, 250.6751556, -217.7781677, 281.6236877, -475.3656616, 468.4532776
4: -167.0168152, 254.3156281, -187.5625153, 285.7385559, -452.7553711, 441.8781128

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8709757, upper bound: 398.8766933
time: 0.79 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8707659, upper bound: 398.8770109
time: 0.81 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -148.8428497, 242.8462067, -162.5143433, 269.2449036, -418.0877075, 405.3605042
1: -163.5882111, 215.5297394, -178.8953094, 238.7306671, -402.3188782, 394.4249878
2: -163.5106354, 219.1772308, -178.8245087, 243.0179749, -406.5286255, 398.0017395
3: -192.6372375, 248.3060455, -211.5433044, 274.9932861, -467.6305237, 459.8493042
4: -166.1027527, 251.9715881, -182.0640717, 278.9735107, -445.0762329, 434.0356445

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A1_A2_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8744040, upper bound: 398.8911555
time: 0.86 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2_A1_B2

### Relational analysis result of NS_B1_A1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8744040, upper bound: 398.8911555
time: 0.83 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -152.7680054, 251.1410828, -160.2622986, 266.2265930, -418.9945374, 411.4032898
1: -167.8907318, 222.7207489, -176.4176178, 235.9544373, -403.8451538, 399.1383667
2: -167.9606628, 226.4134827, -176.3856354, 240.1705780, -408.1312256, 402.7990723
3: -198.0509338, 256.6034546, -208.7162018, 271.7940063, -469.8448486, 465.3196411
4: -170.7062988, 260.1896362, -179.6582489, 275.6768188, -446.3831177, 439.8479004

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_A1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A1_A2_B2_A2_A1

### Relational analysis result of NS_B1_A1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8688651, upper bound: 398.8881311
time: 0.89 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2_A2_A2

### Relational analysis result of NS_B1_A1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8750292, upper bound: 398.8912387
time: 0.86 seconds

## BFS NS instance: NS_B1_A1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -167.9532776, 274.1008301, -166.8633423, 271.7115784, -439.6647644, 440.9641724
1: -184.5478210, 243.2079163, -183.3439789, 241.2205200, -425.7683105, 426.5518799
2: -184.4354706, 247.6372070, -183.1621704, 245.6676941, -430.1031494, 430.7993774
3: -217.2633972, 280.2965088, -215.7885284, 277.9785767, -495.2419739, 496.0850220
4: -187.0465240, 284.5088806, -185.7345123, 282.1824341, -469.2289429, 470.2433472

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A2_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8917500, upper bound: 398.8917500
time: 0.81 seconds

## Relational analysis of NS_B1_A1_A2_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8917500, upper bound: 398.8917516
time: 1.25 seconds

## BFS NS instance: NS_B1_A1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -165.1194916, 270.0919189, -168.5753326, 275.8750916, -440.9945679, 438.6672363
1: -181.4104156, 239.5519257, -185.1636353, 244.7689056, -426.1793213, 424.7155762
2: -181.3399963, 243.9050598, -185.1322327, 249.2006226, -430.5405579, 429.0372925
3: -213.6298828, 276.0788879, -218.1159515, 282.0845032, -495.7143860, 494.1948242
4: -183.9579620, 280.1637573, -187.8461761, 286.2007751, -470.1587219, 468.0099487

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A2_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8917516, upper bound: 398.8917500
time: 0.94 seconds

## Relational analysis of NS_B1_A1_A2_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8917516, upper bound: 398.8917516
time: 1.11 seconds

## BFS NS instance: NS_B1_A1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -161.2266846, 266.9071960, -166.8633423, 271.7115784, -432.9381409, 433.7705383
1: -177.4663391, 236.6997681, -183.3439789, 241.2205200, -418.6867676, 420.0437622
2: -177.4091339, 241.0200500, -183.1621704, 245.6676941, -423.0768127, 424.1822205
3: -209.8498840, 272.6787109, -215.7885284, 277.9785767, -487.8284607, 488.4671631
4: -180.5965576, 276.6629028, -185.7345123, 282.1824341, -462.7789917, 462.3973694

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A2_B1_A2_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8865633, upper bound: 398.8939125
time: 1.31 seconds

## Relational analysis of NS_B1_A1_A2_B1_A2_B1_A2

### Relational analysis result of NS_B1_A1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8878031, upper bound: 398.8939755
time: 0.89 seconds

## BFS NS instance: NS_B1_A1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -158.9586029, 263.8636475, -168.5753326, 275.8750916, -434.8336792, 432.4389648
1: -174.9703674, 233.8987274, -185.1636353, 244.7689056, -419.7392578, 419.0623474
2: -174.9530487, 238.1539154, -185.1322327, 249.2006226, -424.1535950, 423.2861328
3: -207.0063629, 269.4579163, -218.1159515, 282.0845032, -489.0908813, 487.5738525
4: -178.1729736, 273.3470154, -187.8461761, 286.2007751, -464.3737488, 461.1931763

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_A2_B1_A2_B2_A1

### Relational analysis result of NS_B1_A1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8905313, upper bound: 398.8767736
time: 0.85 seconds

## Relational analysis of NS_B1_A1_A2_B1_A2_B2_A2

### Relational analysis result of NS_B1_A1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8892842, upper bound: 398.8771685
time: 0.87 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -165.3875885, 269.0824585, -162.7719269, 269.7117004, -435.0993042, 431.8543701
1: -181.7046051, 238.8484344, -179.1849213, 239.1481323, -420.8527222, 418.0333557
2: -181.5372467, 243.2973175, -179.1102448, 243.4538879, -424.9911194, 422.4075317
3: -213.8449707, 275.3009338, -211.9000854, 275.4799500, -489.3248901, 487.2009888
4: -184.0416412, 279.5148926, -182.3677216, 279.4620361, -463.5036011, 461.8825684

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_B1

### Relational analysis result of NS_B1_A1_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8917500, upper bound: 398.8905470
time: 0.80 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_A1_B2

### Relational analysis result of NS_B1_A1_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8917500, upper bound: 398.8905470
time: 0.90 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -167.1071930, 273.2313538, -160.5177612, 266.6898499, -433.7970276, 433.7491150
1: -183.5337372, 242.3847046, -176.7049713, 236.3683929, -419.9020691, 419.0896606
2: -183.5088196, 246.8202362, -176.6690216, 240.6018677, -424.1106567, 423.4892273
3: -216.1790314, 279.3923645, -209.0705872, 272.2767029, -488.4556885, 488.4629211
4: -186.1596375, 283.5259705, -179.9594574, 276.1615601, -462.3211060, 463.4854126

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_A2_B2_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8767736, upper bound: 398.8905314
time: 0.81 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8771685, upper bound: 398.8892842
time: 1.55 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -161.2266846, 266.9071960, -159.9381256, 264.2120667, -425.4387207, 426.8453369
1: -177.4663391, 236.6997681, -176.0432739, 234.3921661, -411.8584900, 412.7430420
2: -177.4091339, 241.0200500, -175.9160461, 238.7156830, -416.1248169, 416.9360962
3: -209.8498840, 272.6787109, -208.1122742, 270.0297241, -479.8796082, 480.7909851
4: -180.5965576, 276.6629028, -179.0576324, 273.9863586, -454.5829163, 455.7205200

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A2_B2_A2_B1_A1

### Relational analysis result of NS_B1_A1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8905469, upper bound: 398.8913488
time: 0.93 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2_B1_A2

### Relational analysis result of NS_B1_A1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8905469, upper bound: 398.8913951
time: 1.16 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -158.9586029, 263.8636475, -162.8820343, 270.8371582, -429.7957153, 426.7456055
1: -174.9703674, 233.8987274, -179.2815094, 240.2337646, -415.2041321, 413.1802368
2: -174.9530487, 238.1539154, -179.3040466, 244.5386505, -419.4916992, 417.4579468
3: -207.0063629, 269.4579163, -212.2697296, 276.7997437, -483.8060913, 481.7276306
4: -178.1729736, 273.3470154, -182.7162476, 280.6123047, -458.7852783, 456.0632629

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_A1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A2_B2_A2_B2_B1

### Relational analysis result of NS_B1_A1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8858683, upper bound: 398.8833103
time: 0.88 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2_B2_B2

### Relational analysis result of NS_B1_A1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8913861, upper bound: 398.8913951
time: 0.92 seconds

## BFS NS instance: NS_B1_A2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -370.3743591, 552.7754517, -154.4789124, 254.6170502, -611.1141968, 701.4907837
1: -404.3429565, 505.6080627, -169.9770813, 225.9568634, -617.6701660, 669.2297974
2: -403.4871826, 514.5595703, -169.8003540, 230.1386108, -622.8751831, 677.8446655
3: -470.6140442, 583.7698364, -200.7553406, 260.3443604, -721.2680054, 776.9550171
4: -403.5347595, 589.8770752, -172.7836304, 264.0625305, -658.2861328, 759.4127808

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B1_A2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_A1_A1_B1_B1_B1

### Relational analysis result of NS_B1_A2_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8405622, upper bound: 398.9017437
time: 1.38 seconds

## Relational analysis of NS_B1_A2_A1_A1_B1_B1_B2

### Relational analysis result of NS_B1_A2_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8406865, upper bound: 398.9050041
time: 1.03 seconds

## BFS NS instance: NS_B1_A2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -366.6619568, 546.8342285, -157.8313904, 261.9318542, -614.9177856, 698.9976807
1: -400.2093506, 500.2612000, -173.6621399, 232.4026642, -620.1347046, 667.7887573
2: -399.3562622, 509.0968628, -173.6453552, 236.5879517, -625.3419189, 676.3511963
3: -465.7137451, 577.5889893, -205.4298706, 267.8133545, -723.9813843, 775.6364746
4: -399.3481140, 583.4851685, -176.8878174, 271.3846130, -661.5625610, 757.3435669

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_A1_A1_B1_B2_B1

### Relational analysis result of NS_B1_A2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8495646, upper bound: 398.8884357
time: 1.07 seconds

## Relational analysis of NS_B1_A2_A1_A1_B1_B2_B2

### Relational analysis result of NS_B1_A2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8502246, upper bound: 398.8883002
time: 0.89 seconds

## BFS NS instance: NS_B1_A2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -367.0601196, 546.3156738, -174.2644348, 280.6953125, -633.1869507, 714.0668335
1: -400.5968628, 500.2153931, -191.1528320, 249.6103668, -637.7673340, 684.2947388
2: -399.7536011, 509.1206055, -191.3220978, 254.3036957, -643.3432007, 693.3754272
3: -466.0394897, 577.6259766, -224.7953033, 287.7098999, -744.4431152, 794.2050171
4: -399.5914001, 583.5501709, -193.5199738, 292.4315796, -682.5796509, 773.2302856

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_A1_A1_B2_B1_B1

### Relational analysis result of NS_B1_A2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8372846, upper bound: 398.8601280
time: 0.96 seconds

## Relational analysis of NS_B1_A2_A1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B1_A2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_A1_A1_B2_B1_A1

### Relational analysis result of NS_B1_A2_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8384215, upper bound: 398.8640028
time: 1.11 seconds

## Relational analysis of NS_B1_A2_A1_A1_B2_B1_A2

### Relational analysis result of NS_B1_A2_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8384215, upper bound: 398.8640084
time: 0.88 seconds

## BFS NS instance: NS_B1_A2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -367.6709290, 547.2175903, -184.3855133, 296.8973694, -650.0756836, 725.0607300
1: -401.2790222, 500.9298706, -202.3422394, 264.1164551, -652.9545898, 696.1287842
2: -400.4082336, 509.8492126, -202.4465790, 268.9466858, -658.7229004, 705.2156372
3: -466.8101501, 578.4333496, -238.0636292, 304.4267883, -761.9331055, 808.2297363
4: -400.2497253, 584.4383545, -204.7925568, 309.3838196, -700.2149658, 785.2998657

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A1_A1_B2_B2_A1

### Relational analysis result of NS_B1_A2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8504049, upper bound: 398.8864147
time: 0.80 seconds

## Relational analysis of NS_B1_A2_A1_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A1_A1_B2_B2_A1

### Relational analysis result of NS_B1_A2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8061439, upper bound: 398.8840741
time: 1.76 seconds

## Relational analysis of NS_B1_A2_A1_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_A1_A1_B2_B2_B1

### Relational analysis result of NS_B1_A2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8499079, upper bound: 398.8862838
time: 1.07 seconds

## Relational analysis of NS_B1_A2_A1_A1_B2_B2_B2

### Relational analysis result of NS_B1_A2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8501177, upper bound: 398.8805710
time: 0.73 seconds

## BFS NS instance: NS_B1_A2_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -361.4888000, 539.1544800, -156.7750549, 258.6855774, -606.3490601, 689.8836670
1: -394.6794739, 494.4166870, -172.5573578, 229.6258850, -612.0513916, 660.5227051
2: -393.6528320, 503.2125244, -172.3812256, 233.9074707, -617.4165039, 668.8948975
3: -459.5310059, 570.8867188, -203.9600677, 264.5612793, -714.6423340, 767.1995239
4: -394.0489197, 576.3262329, -175.4993896, 268.3612366, -653.3070679, 748.4982910

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_A1_A2_B1_A1_B1

### Relational analysis result of NS_B1_A2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8597428, upper bound: 398.9091543
time: 0.90 seconds

## Relational analysis of NS_B1_A2_A1_A2_B1_A1_B2

### Relational analysis result of NS_B1_A2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8635669, upper bound: 398.9105426
time: 0.80 seconds

## BFS NS instance: NS_B1_A2_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -365.1779785, 546.1354980, -157.9800262, 261.1013184, -612.7402954, 698.2928467
1: -398.8798218, 500.2301636, -173.8676758, 231.6093750, -618.4266357, 667.8024902
2: -397.7702942, 509.1393127, -173.7283783, 235.8585815, -623.6131592, 676.4309692
3: -464.6615906, 577.5059814, -205.4851837, 266.8147583, -722.2304688, 775.5183105
4: -398.4960938, 583.2584229, -176.8605042, 270.6324463, -660.2291870, 757.0566406

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_A1_A2_B1_A2_B1

### Relational analysis result of NS_B1_A2_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8643831, upper bound: 398.9105097
time: 1.61 seconds

## Relational analysis of NS_B1_A2_A1_A2_B1_A2_B2

### Relational analysis result of NS_B1_A2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8643525, upper bound: 398.8895449
time: 0.96 seconds

## BFS NS instance: NS_B1_A2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -359.3337708, 535.0977173, -178.3736267, 287.4222717, -632.5686646, 706.8870850
1: -392.2817078, 490.8361816, -195.7200012, 255.6912842, -636.0744629, 679.6286011
2: -391.2326965, 499.5905762, -195.8648987, 260.5161133, -641.8881836, 688.4142456
3: -456.5899963, 566.7991943, -230.1878662, 294.7182312, -742.3564453, 788.9722290
4: -391.4625549, 572.1889648, -198.1467743, 299.4332886, -681.9015503, 766.6238403

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_A1_A2_B2_A1_B1

### Relational analysis result of NS_B1_A2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8629835, upper bound: 398.8864757
time: 1.78 seconds

## Relational analysis of NS_B1_A2_A1_A2_B2_A1_B2

### Relational analysis result of NS_B1_A2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8629878, upper bound: 398.8813791
time: 0.91 seconds

## BFS NS instance: NS_B1_A2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -363.5958862, 543.1818848, -179.5149689, 289.7413940, -639.4861450, 716.3329468
1: -397.1154175, 497.7056885, -196.9798126, 257.5881042, -643.0385132, 687.9238281
2: -395.9917603, 506.5915527, -197.1445312, 262.4199219, -648.6940308, 696.9494019
3: -462.5126953, 574.6210938, -231.6984558, 296.9069824, -750.6342163, 798.5141602
4: -396.6145630, 580.2912598, -199.4725800, 301.6571350, -689.4257812, 776.2772827

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_A1_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8521413, upper bound: 398.8640238
time: 0.85 seconds

## Relational analysis of NS_B1_A2_A1_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8640036, upper bound: 398.8875607
time: 0.96 seconds

## BFS NS instance: NS_B1_A2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -379.6077271, 569.6266479, -154.7386169, 255.0899353, -621.1700439, 719.7757568
1: -414.7713318, 520.5842285, -170.2688599, 226.3798828, -628.7645874, 685.3756104
2: -413.8284302, 529.7232056, -170.0887451, 230.5727081, -633.9904175, 694.3001099
3: -483.3833923, 600.8834839, -201.1167297, 260.8375549, -734.8314819, 795.5105591
4: -414.5258789, 607.3905640, -173.0892487, 264.5568848, -670.1875610, 777.9824829

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A2_A1_B1_B1_A1

### Relational analysis result of NS_B1_A2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8521628, upper bound: 398.8957681
time: 0.89 seconds

## Relational analysis of NS_B1_A2_A2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_A2_A1_B1_B1_B1

### Relational analysis result of NS_B1_A2_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8642422, upper bound: 398.9015026
time: 0.99 seconds

## Relational analysis of NS_B1_A2_A2_A1_B1_B1_B2

### Relational analysis result of NS_B1_A2_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8643850, upper bound: 398.9049654
time: 1.01 seconds

## BFS NS instance: NS_B1_A2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -375.8770752, 563.7370605, -158.0704498, 262.3685303, -624.9489746, 717.3109131
1: -410.6222839, 515.2498779, -173.9314423, 232.7933960, -631.2050171, 683.9241333
2: -409.6811218, 524.2864380, -173.9107819, 236.9901276, -636.4319458, 692.8050537
3: -478.4651489, 594.7066650, -205.7627869, 268.2694397, -737.5023193, 794.1715698
4: -410.3206482, 601.0079346, -177.1710510, 271.8426819, -673.4199219, 775.8942261

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_A2_A1_B1_B2_B1

### Relational analysis result of NS_B1_A2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8638558, upper bound: 398.8885386
time: 1.32 seconds

## Relational analysis of NS_B1_A2_A2_A1_B1_B2_B2

### Relational analysis result of NS_B1_A2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8644972, upper bound: 398.8883623
time: 1.05 seconds

## BFS NS instance: NS_B1_A2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -376.3793945, 563.2750244, -174.5328522, 281.1800232, -643.3634644, 732.4996338
1: -411.1383972, 515.2647705, -191.4541168, 250.0276794, -648.9926758, 700.5742188
2: -410.1897278, 524.3527222, -191.6191406, 254.7433472, -654.5726318, 709.9580688
3: -478.9483948, 594.8383179, -225.1613007, 288.2007446, -758.1770020, 812.9223633
4: -410.7136230, 601.2125854, -193.8288727, 292.9226990, -694.6478271, 791.9586792

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A2_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_A2_A1_B2_B1_B1

### Relational analysis result of NS_B1_A2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8596693, upper bound: 398.8607878
time: 0.95 seconds

## Relational analysis of NS_B1_A2_A2_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_A2_A1_B2_B1_A1

### Relational analysis result of NS_B1_A2_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8600052, upper bound: 398.8602228
time: 1.26 seconds

## Relational analysis of NS_B1_A2_A2_A1_B2_B1_A2

### Relational analysis result of NS_B1_A2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8600052, upper bound: 398.8690146
time: 1.05 seconds

## BFS NS instance: NS_B1_A2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -376.9158020, 564.0800781, -184.6097412, 297.3053589, -660.0502930, 743.3000488
1: -411.7383728, 515.9526978, -202.5921021, 264.4699707, -663.9825439, 712.2658081
2: -410.7675476, 525.0591431, -202.6967316, 269.3192749, -669.7642212, 721.6759644
3: -479.6352844, 595.6164551, -238.3652954, 304.8326416, -775.4580078, 826.7811890
4: -411.2931213, 602.0433350, -205.0497284, 309.7965088, -712.0737305, 803.8997803

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A2_A1_B2_B2_A1

### Relational analysis result of NS_B1_A2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8641320, upper bound: 398.8865421
time: 1.03 seconds

## Relational analysis of NS_B1_A2_A2_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_B1_A2_A2_A1_B2_B2_A1

### Relational analysis result of NS_B1_A2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8184894, upper bound: 398.8841957
time: 0.95 seconds

## Relational analysis of NS_B1_A2_A2_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_A2_A1_B2_B2_B1

### Relational analysis result of NS_B1_A2_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8642432, upper bound: 398.8864345
time: 0.93 seconds

## Relational analysis of NS_B1_A2_A2_A1_B2_B2_B2

### Relational analysis result of NS_B1_A2_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8643984, upper bound: 398.8806098
time: 1.00 seconds

## BFS NS instance: NS_B1_A2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -371.0113525, 556.3269043, -157.0335541, 259.1543274, -616.6763916, 708.4355469
1: -405.4329224, 509.8593140, -172.8480530, 230.0449677, -623.4431152, 677.1374512
2: -404.2937622, 518.8746338, -172.6678925, 234.3457642, -628.8201294, 685.8591309
3: -472.7577209, 588.5453491, -204.3228912, 265.0497131, -728.6303101, 786.2849731
4: -405.4259644, 594.4384155, -175.8031006, 268.8520508, -665.5703735, 767.6552124

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_A2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_A2_A2_B1_A1_B1

### Relational analysis result of NS_B1_A2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8770949, upper bound: 398.9090225
time: 0.83 seconds

## Relational analysis of NS_B1_A2_A2_A2_B1_A1_B2

### Relational analysis result of NS_B1_A2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8770467, upper bound: 398.8881524
time: 0.87 seconds

## BFS NS instance: NS_B1_A2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -373.0596619, 560.8522949, -158.2331848, 261.5621033, -621.4588013, 714.4099731
1: -407.8164368, 513.3724976, -174.1520538, 232.0212708, -628.0358276, 682.0827026
2: -406.6366577, 522.4588623, -174.0092163, 236.2819672, -633.2669067, 691.0133667
3: -475.7110901, 592.5132446, -205.8365326, 267.2948608, -734.0484009, 791.9213257
4: -408.0168152, 598.6420288, -177.1585846, 271.1149597, -670.6079712, 773.4916382

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_A2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B1_A2_A2_A2_B1_A2_B1

### Relational analysis result of NS_B1_A2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8786176, upper bound: 398.9105942
time: 0.95 seconds

## Relational analysis of NS_B1_A2_A2_A2_B1_A2_B2

### Relational analysis result of NS_B1_A2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8785603, upper bound: 398.8896031
time: 0.94 seconds

## BFS NS instance: NS_B1_A2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -369.0555725, 552.6056519, -178.6524048, 287.9208374, -643.1304321, 725.8038940
1: -403.2678528, 506.5930176, -196.0329437, 256.1319275, -647.7215576, 696.5690308
2: -402.0958557, 515.5739136, -196.1735077, 260.9691467, -653.5365601, 705.7208862
3: -470.1093445, 584.8215942, -230.5705872, 295.2238159, -756.6510620, 808.4423218
4: -403.0974121, 590.6898804, -198.4666290, 299.9412537, -694.4441528, 786.1864624

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_A2_A2_B2_A1_B1

### Relational analysis result of NS_B1_A2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8799820, upper bound: 398.8703388
time: 0.78 seconds

## Relational analysis of NS_B1_A2_A2_A2_B2_A1_B2

### Relational analysis result of NS_B1_A2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8810380, upper bound: 398.8881870
time: 1.10 seconds

## BFS NS instance: NS_B1_A2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -371.6136780, 558.1476440, -179.7827301, 290.2216187, -648.3764038, 732.6964722
1: -406.2141418, 511.0833130, -197.2807465, 258.0054321, -652.8322144, 702.4527588
2: -405.0126953, 520.1481323, -197.4412994, 262.8567505, -658.5257568, 711.7942505
3: -473.7716370, 589.9020996, -232.0649567, 297.3936462, -762.6655884, 815.2107544
4: -406.3240356, 595.9624634, -199.7802734, 302.1458740, -700.0095825, 792.9978027

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_B1_A2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A2_A2_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8808327, upper bound: 398.8710670
time: 1.17 seconds

## Relational analysis of NS_B1_A2_A2_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8817347, upper bound: 398.8881808
time: 1.11 seconds

## BFS NS instance: NS_B2_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -154.4789124, 254.6170502, -370.3743591, 552.7754517, -701.4907227, 611.1141968
1: -169.9770813, 225.9568634, -404.3429565, 505.6080627, -669.2297974, 617.6701660
2: -169.8003540, 230.1386108, -403.4871826, 514.5595703, -677.8446655, 622.8752441
3: -200.7553406, 260.3443604, -470.6140442, 583.7698364, -776.9550171, 721.2680664
4: -172.7836304, 264.0625305, -403.5348206, 589.8770752, -759.4127808, 658.2861938

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_B1_B1_A1_A1_A1

### Relational analysis result of NS_B2_A1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9017437, upper bound: 398.8405622
time: 1.08 seconds

## Relational analysis of NS_B2_A1_B1_B1_A1_A1_A2

### Relational analysis result of NS_B2_A1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9050041, upper bound: 398.8406865
time: 1.04 seconds

## BFS NS instance: NS_B2_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -157.8313904, 261.9318542, -366.6619568, 546.8342285, -698.9976807, 614.9177856
1: -173.6621399, 232.4026642, -400.2093506, 500.2612000, -667.7887573, 620.1346436
2: -173.6453552, 236.5879517, -399.3562622, 509.0968628, -676.3511353, 625.3419189
3: -205.4298706, 267.8133545, -465.7137451, 577.5889893, -775.6364746, 723.9813843
4: -176.8878174, 271.3846130, -399.3481140, 583.4851685, -757.3435669, 661.5625610

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B1_B1_A1_A2_A1

### Relational analysis result of NS_B2_A1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8884357, upper bound: 398.8495646
time: 1.03 seconds

## Relational analysis of NS_B2_A1_B1_B1_A1_A2_A2

### Relational analysis result of NS_B2_A1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8883002, upper bound: 398.8502246
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -174.2644348, 280.6953125, -367.0601501, 546.3156738, -714.0668335, 633.1869507
1: -191.1528320, 249.6103668, -400.5968628, 500.2154236, -684.2947998, 637.7672729
2: -191.3220978, 254.3036957, -399.7536011, 509.1206055, -693.3754272, 643.3432007
3: -224.7953033, 287.7098999, -466.0394897, 577.6259766, -794.2050171, 744.4431763
4: -193.5199738, 292.4315796, -399.5914001, 583.5501709, -773.2302856, 682.5796509

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_B1_B1_A2_A1_A1

### Relational analysis result of NS_B2_A1_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8601280, upper bound: 398.8372846
time: 0.86 seconds

## Relational analysis of NS_B2_A1_B1_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_B2_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_B2_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_B2_A1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B1_B1_A2_A1_B1

### Relational analysis result of NS_B2_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8640028, upper bound: 398.8384215
time: 0.85 seconds

## Relational analysis of NS_B2_A1_B1_B1_A2_A1_B2

### Relational analysis result of NS_B2_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8640028, upper bound: 398.8384215
time: 4.47 seconds

## BFS NS instance: NS_B2_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -184.3855133, 296.8973694, -367.6709290, 547.2175903, -725.0606689, 650.0756836
1: -202.3422394, 264.1164551, -401.2790222, 500.9298706, -696.1288452, 652.9546509
2: -202.4465790, 268.9466858, -400.4082336, 509.8492126, -705.2156372, 658.7229004
3: -238.0636292, 304.4267883, -466.8101501, 578.4333496, -808.2297363, 761.9331055
4: -204.7925568, 309.3838196, -400.2497253, 584.4383545, -785.2999268, 700.2149658

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_B1_B1_A2_A2_B1

### Relational analysis result of NS_B2_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8864147, upper bound: 398.8504049
time: 1.00 seconds

## Relational analysis of NS_B2_A1_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B1_B1_A2_A2_B1

### Relational analysis result of NS_B2_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8840741, upper bound: 398.8061439
time: 0.98 seconds

## Relational analysis of NS_B2_A1_B1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_B1_B1_A2_A2_A1

### Relational analysis result of NS_B2_A1_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8862838, upper bound: 398.8499079
time: 0.77 seconds

## Relational analysis of NS_B2_A1_B1_B1_A2_A2_A2

### Relational analysis result of NS_B2_A1_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8805710, upper bound: 398.8501177
time: 0.85 seconds

## BFS NS instance: NS_B2_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -156.7750549, 258.6855774, -361.4888000, 539.1544800, -689.8836060, 606.3490601
1: -172.5573578, 229.6258850, -394.6794739, 494.4166870, -660.5227051, 612.0513916
2: -172.3812256, 233.9074707, -393.6528320, 503.2125244, -668.8948975, 617.4165039
3: -203.9600677, 264.5612793, -459.5310059, 570.8867188, -767.1995239, 714.6423340
4: -175.4993896, 268.3612366, -394.0489197, 576.3262329, -748.4982910, 653.3070679

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B1_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9091543, upper bound: 398.8597428
time: 1.21 seconds

## Relational analysis of NS_B2_A1_B1_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9105426, upper bound: 398.8635669
time: 0.76 seconds

## BFS NS instance: NS_B2_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -157.9800262, 261.1013184, -365.1779480, 546.1354980, -698.2927856, 612.7402954
1: -173.8676758, 231.6093750, -398.8798218, 500.2301025, -667.8024292, 618.4265747
2: -173.7283783, 235.8585815, -397.7702637, 509.1392517, -676.4309082, 623.6131592
3: -205.4851837, 266.8147583, -464.6615906, 577.5059814, -775.5183105, 722.2304688
4: -176.8605042, 270.6324463, -398.4960938, 583.2583618, -757.0566406, 660.2291870

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_B1_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9105097, upper bound: 398.8643831
time: 0.86 seconds

## Relational analysis of NS_B2_A1_B1_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8895449, upper bound: 398.8643525
time: 1.53 seconds

## BFS NS instance: NS_B2_A1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -178.3736267, 287.4222717, -359.3337708, 535.0977173, -706.8871460, 632.5686646
1: -195.7200012, 255.6912842, -392.2817078, 490.8361816, -679.6286011, 636.0743408
2: -195.8648987, 260.5161133, -391.2326660, 499.5905762, -688.4141846, 641.8881836
3: -230.1878662, 294.7182312, -456.5899963, 566.7991943, -788.9722900, 742.3563843
4: -198.1467743, 299.4332886, -391.4625244, 572.1889648, -766.6238403, 681.9015503

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_B1_B2_A2_B1_A1

### Relational analysis result of NS_B2_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8864757, upper bound: 398.8629835
time: 1.12 seconds

## Relational analysis of NS_B2_A1_B1_B2_A2_B1_A2

### Relational analysis result of NS_B2_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8813791, upper bound: 398.8629878
time: 1.00 seconds

## BFS NS instance: NS_B2_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -179.5149689, 289.7413940, -363.5959167, 543.1818848, -716.3329468, 639.4860840
1: -196.9798126, 257.5881042, -397.1154175, 497.7056885, -687.9237671, 643.0385132
2: -197.1445312, 262.4199219, -395.9917603, 506.5915527, -696.9494019, 648.6940308
3: -231.6984558, 296.9069824, -462.5126953, 574.6210938, -798.5141602, 750.6342773
4: -199.4725800, 301.6571350, -396.6145020, 580.2911987, -776.2772217, 689.4257202

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B1_B2_A2_B2_A1

### Relational analysis result of NS_B2_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8640238, upper bound: 398.8521413
time: 1.13 seconds

## Relational analysis of NS_B2_A1_B1_B2_A2_B2_A2

### Relational analysis result of NS_B2_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8875607, upper bound: 398.8640036
time: 0.87 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -154.7386169, 255.0899353, -379.6077271, 569.6267090, -719.7758179, 621.1700439
1: -170.2688599, 226.3798828, -414.7713013, 520.5842285, -685.3756104, 628.7646484
2: -170.0887451, 230.5727081, -413.8284912, 529.7232056, -694.3001099, 633.9904785
3: -201.1167297, 260.8375549, -483.3834229, 600.8834839, -795.5104980, 734.8315430
4: -173.0892487, 264.5568848, -414.5258789, 607.3905640, -777.9824829, 670.1875610

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8957681, upper bound: 398.8521628
time: 0.87 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9015026, upper bound: 398.8642422
time: 0.96 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A1_A2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9049654, upper bound: 398.8643850
time: 0.98 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -158.0704498, 262.3685303, -375.8771057, 563.7369995, -717.3109131, 624.9490356
1: -173.9314423, 232.7933960, -410.6222839, 515.2499390, -683.9241333, 631.2050781
2: -173.9107819, 236.9901276, -409.6811218, 524.2864380, -692.8050537, 636.4318848
3: -205.7627869, 268.2694397, -478.4651794, 594.7066650, -794.1715698, 737.5023193
4: -177.1710510, 271.8426819, -410.3206482, 601.0078735, -775.8942261, 673.4198608

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8885386, upper bound: 398.8638558
time: 0.95 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A2_A2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8883623, upper bound: 398.8644972
time: 1.06 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -174.5328522, 281.1800232, -376.3793945, 563.2750244, -732.4996338, 643.3635254
1: -191.4541168, 250.0276794, -411.1383972, 515.2647705, -700.5741577, 648.9926758
2: -191.6191406, 254.7433472, -410.1897278, 524.3527222, -709.9580688, 654.5726318
3: -225.1613007, 288.2007446, -478.9483948, 594.8383179, -812.9223633, 758.1770020
4: -193.8288727, 292.9226990, -410.7136230, 601.2125854, -791.9586792, 694.6478271

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_A1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8607878, upper bound: 398.8596693
time: 1.14 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8602228, upper bound: 398.8600052
time: 0.87 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A1_B2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8602228, upper bound: 398.8600052
time: 1.03 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -184.6097412, 297.3053589, -376.9158020, 564.0800781, -743.3000488, 660.0502930
1: -202.5921021, 264.4699707, -411.7383728, 515.9526978, -712.2657471, 663.9824829
2: -202.6967316, 269.3192749, -410.7675476, 525.0591431, -721.6759644, 669.7642212
3: -238.3652954, 304.8326416, -479.6352844, 595.6164551, -826.7811890, 775.4580078
4: -205.0497284, 309.7965088, -411.2931213, 602.0433350, -803.8997803, 712.0737305

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8865421, upper bound: 398.8641320
time: 1.03 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8841957, upper bound: 398.8184894
time: 0.84 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_A1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8864345, upper bound: 398.8642432
time: 0.80 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A2_A2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8806098, upper bound: 398.8643984
time: 0.78 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -157.0335541, 259.1543274, -371.0113525, 556.3269043, -708.4355469, 616.6763916
1: -172.8480530, 230.0449677, -405.4329224, 509.8593140, -677.1374512, 623.4431152
2: -172.6678925, 234.3457642, -404.2937622, 518.8746338, -685.8591309, 628.8201294
3: -204.3228912, 265.0497131, -472.7577209, 588.5453491, -786.2849731, 728.6303101
4: -175.8031006, 268.8520508, -405.4259644, 594.4384155, -767.6552124, 665.5703735

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_B2_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9090225, upper bound: 398.8770949
time: 0.92 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8881524, upper bound: 398.8770467
time: 0.96 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -158.2331848, 261.5621033, -373.0596619, 560.8522949, -714.4099731, 621.4588013
1: -174.1520538, 232.0212708, -407.8164368, 513.3724976, -682.0827637, 628.0357666
2: -174.0092163, 236.2819672, -406.6366577, 522.4588623, -691.0133057, 633.2669067
3: -205.8365326, 267.2948608, -475.7110901, 592.5132446, -791.9213257, 734.0484009
4: -177.1585846, 271.1149597, -408.0168152, 598.6420288, -773.4916382, 670.6079712

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_B2_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.9105942, upper bound: 398.8786176
time: 0.82 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8896031, upper bound: 398.8785603
time: 1.34 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -178.6524048, 287.9208374, -369.0555725, 552.6056519, -725.8038940, 643.1303711
1: -196.0329437, 256.1319275, -403.2678528, 506.5930176, -696.5690918, 647.7215576
2: -196.1735077, 260.9691467, -402.0958557, 515.5739136, -705.7208862, 653.5365601
3: -230.5705872, 295.2238159, -470.1093445, 584.8215942, -808.4423218, 756.6511230
4: -198.4666290, 299.9412537, -403.0974121, 590.6898804, -786.1864624, 694.4441528

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B2_B2_A2_B1_A1

### Relational analysis result of NS_B2_A1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8703388, upper bound: 398.8799820
time: 0.81 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_B1_A2

### Relational analysis result of NS_B2_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8881871, upper bound: 398.8810380
time: 0.81 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -179.7827301, 290.2216187, -371.6136475, 558.1475830, -732.6964111, 648.3763428
1: -197.2807465, 258.0054321, -406.2141418, 511.0833130, -702.4527588, 652.8322144
2: -197.4412994, 262.8567505, -405.0126343, 520.1481323, -711.7942505, 658.5256958
3: -232.0649567, 297.3936462, -473.7716370, 589.9020996, -815.2107544, 762.6654663
4: -199.7802734, 302.1458740, -406.3239746, 595.9624634, -792.9978027, 700.0095215

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B2_A1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_B2_A1_B2_B2_A2_B2_A1

### Relational analysis result of NS_B2_A1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8710670, upper bound: 398.8808327
time: 0.75 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_B2_A2

### Relational analysis result of NS_B2_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8881808, upper bound: 398.8817347
time: 1.00 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -359.1047974, 535.0496216, -377.5628052, 569.2854614, -904.9832153, 888.9394531
1: -392.1916809, 489.7494812, -412.7815552, 520.2795410, -889.4906616, 879.4539185
2: -391.2513123, 498.1222229, -411.6668396, 529.4323730, -899.0869141, 888.2371216
3: -456.5638428, 565.3795776, -481.6786804, 600.4142456, -1036.1557617, 1026.1408691
4: -391.3735962, 571.4877319, -413.0681152, 606.6361084, -982.2768555, 968.4276733

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A1_B2_A1_A1_B1

### Relational analysis result of NS_B2_A2_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8131496, upper bound: 398.8783316
time: 0.85 seconds

## Relational analysis of NS_B2_A2_A1_B2_A1_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8131496, upper bound: 398.8805966
time: 1.25 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -355.8411255, 532.1540527, -381.7806702, 576.3164062, -909.4552002, 890.3541870
1: -388.8265991, 487.5883789, -417.4514771, 526.8095093, -893.4423218, 882.1165161
2: -387.7518311, 495.9043884, -416.3379517, 536.0493774, -903.1318970, 890.8949585
3: -453.2017517, 562.8571167, -487.2684021, 607.9335938, -1041.0325928, 1029.3537598
4: -388.4735107, 568.4777222, -417.8563538, 614.0875854, -987.4274292, 970.3944702

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A1_B2_A1_A2_B1

### Relational analysis result of NS_B2_A2_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8131496, upper bound: 398.8783316
time: 1.18 seconds

## Relational analysis of NS_B2_A2_A1_B2_A1_A2_B2

### Relational analysis result of NS_B2_A2_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8131496, upper bound: 398.8807346
time: 1.20 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -372.2371826, 555.0605469, -377.5994873, 569.3382568, -918.3499756, 909.4904785
1: -406.3552246, 507.7509766, -412.8216858, 520.3284912, -903.9367065, 897.9744873
2: -405.4764099, 516.7338257, -411.7062073, 529.4838257, -913.6071167, 907.3226929
3: -472.8561707, 586.2639160, -481.7247314, 600.4702148, -1052.8468018, 1047.1690674
4: -405.4433594, 592.4120483, -413.1076355, 606.6948853, -996.2979126, 989.7935181

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A1_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8602472, upper bound: 398.8789031
time: 0.86 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8602472, upper bound: 398.8793485
time: 0.84 seconds

## BFS NS instance: NS_B2_A2_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -369.9821777, 554.3228149, -381.7839661, 576.3209839, -923.8959351, 913.0491943
1: -404.1330566, 507.4989014, -417.4550171, 526.8139038, -909.1013184, 902.5336914
2: -403.1123962, 516.4439087, -416.3414307, 536.0538940, -918.8395386, 911.8950195
3: -470.9189453, 585.8983154, -487.2724609, 607.9384766, -1059.0844727, 1052.5498047
4: -403.8274536, 591.5707397, -417.8598022, 614.0927734, -1002.6630249, 993.8775635

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A1_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8618052, upper bound: 398.8789031
time: 1.07 seconds

## Relational analysis of NS_B2_A2_A1_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8618052, upper bound: 398.8808937
time: 0.99 seconds

## BFS NS instance: NS_B2_A2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -367.9607239, 551.4232178, -377.8157654, 569.7274780, -914.5810547, 906.7163086
1: -402.2008057, 504.2780762, -413.0641174, 520.6706543, -900.0713501, 895.0824585
2: -401.1758728, 512.8526001, -411.9478149, 529.8298950, -909.6979980, 904.2268066
3: -468.8801270, 581.9733276, -482.0223389, 600.8615112, -1049.0935059, 1044.1130371
4: -401.9493103, 588.4746094, -413.3624878, 607.0930176, -993.6539917, 986.4010620

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A2_A1_B2_A1_B1

### Relational analysis result of NS_B2_A2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8200840, upper bound: 398.8783631
time: 0.85 seconds

## Relational analysis of NS_B2_A2_A2_A1_B2_A1_B2

### Relational analysis result of NS_B2_A2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8200840, upper bound: 398.8806619
time: 0.93 seconds

## BFS NS instance: NS_B2_A2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -364.8708801, 548.7542725, -382.0289001, 576.7522583, -919.2324829, 908.3615723
1: -399.0309753, 502.4246521, -417.7287598, 527.1953125, -904.2208862, 898.1044312
2: -397.8590698, 510.9677124, -416.6140137, 536.4409790, -913.9329224, 907.2728882
3: -465.7769165, 579.7923584, -487.6061401, 608.3746338, -1054.2232666, 1047.7135010
4: -399.2795410, 585.8625488, -418.1456299, 614.5377197, -998.9916382, 988.7741089

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A2_A1_B2_A2_B1

### Relational analysis result of NS_B2_A2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8200840, upper bound: 398.8783631
time: 0.74 seconds

## Relational analysis of NS_B2_A2_A2_A1_B2_A2_B2

### Relational analysis result of NS_B2_A2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8200840, upper bound: 398.8807988
time: 0.91 seconds

## BFS NS instance: NS_B2_A2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -376.5298462, 567.2085571, -382.4596558, 574.0285645, -928.4281616, 926.8964233
1: -411.6247559, 518.5527954, -417.9093018, 524.4977417, -914.3273926, 913.9763184
2: -410.5067444, 527.6518555, -416.9402771, 533.7221069, -924.0048218, 923.6708374
3: -480.2465820, 598.4447632, -487.0619507, 605.3965454, -1065.8498535, 1065.3273926
4: -411.8572388, 604.6469116, -417.6451416, 611.9722900, -1008.7848511, 1006.8978882

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_B2_A2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A2_A2_B2_B1_A1

### Relational analysis result of NS_B2_A2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8782432, upper bound: 398.8790301
time: 0.95 seconds

## Relational analysis of NS_B2_A2_A2_A2_B2_B1_A2

### Relational analysis result of NS_B2_A2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8782432, upper bound: 398.8790301
time: 0.98 seconds

## BFS NS instance: NS_B2_A2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -380.7317200, 574.2249146, -380.4638977, 573.6807251, -932.3838501, 932.7343140
1: -416.2763977, 525.0709229, -415.9718018, 524.6908569, -919.3936768, 919.4669800
2: -415.1612549, 534.2542725, -414.8568420, 533.9025879, -929.1139526, 929.2296143
3: -485.8170166, 605.9503174, -485.4714661, 605.5255737, -1071.7889404, 1071.9257812
4: -416.6303406, 612.0791016, -416.3242798, 611.6270142, -1013.4070435, 1013.5478516

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B2_A2_A2_A2_B2_B2_A1

### Relational analysis result of NS_B2_A2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8782432, upper bound: 398.8795060
time: 0.81 seconds

## Relational analysis of NS_B2_A2_A2_A2_B2_B2_A2

### Relational analysis result of NS_B2_A2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8782432, upper bound: 398.8810410
time: 0.88 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.56 seconds
NS_B1_A1_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8742491, upper bound: 398.8916180
NS_B1_A1_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8742491, upper bound: 398.8916180
NS_B1_A1_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8742491, upper bound: 398.8916204
NS_B1_A1_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8742491, upper bound: 398.8916204
NS_B1_A1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8750163, upper bound: 398.8903971
NS_B1_A1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8750163, upper bound: 398.8903971
NS_B1_A1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8750163, upper bound: 398.8912387
NS_B1_A1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8750163, upper bound: 398.8912387
NS_B1_A1_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8543259, upper bound: 398.8939104
NS_B1_A1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8616591, upper bound: 398.8938938
NS_B1_A1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8709757, upper bound: 398.8766933
NS_B1_A1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8707659, upper bound: 398.8770109
NS_B1_A1_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8744040, upper bound: 398.8911555
NS_B1_A1_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8744040, upper bound: 398.8911555
NS_B1_A1_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8688651, upper bound: 398.8881311
NS_B1_A1_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8750292, upper bound: 398.8912387
NS_B1_A1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8917500, upper bound: 398.8917500
NS_B1_A1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8917500, upper bound: 398.8917516
NS_B1_A1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8917516, upper bound: 398.8917500
NS_B1_A1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8917516, upper bound: 398.8917516
NS_B1_A1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8865633, upper bound: 398.8939125
NS_B1_A1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8878031, upper bound: 398.8939755
NS_B1_A1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8905313, upper bound: 398.8767736
NS_B1_A1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8892842, upper bound: 398.8771685
NS_B1_A1_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8917500, upper bound: 398.8905470
NS_B1_A1_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8917500, upper bound: 398.8905470
NS_B1_A1_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8767736, upper bound: 398.8905314
NS_B1_A1_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8771685, upper bound: 398.8892842
NS_B1_A1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8905469, upper bound: 398.8913488
NS_B1_A1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8905469, upper bound: 398.8913951
NS_B1_A1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8858683, upper bound: 398.8833103
NS_B1_A1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8913861, upper bound: 398.8913951
NS_B1_A2_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8405622, upper bound: 398.9017437
NS_B1_A2_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8406865, upper bound: 398.9050041
NS_B1_A2_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8495646, upper bound: 398.8884357
NS_B1_A2_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8502246, upper bound: 398.8883002
NS_B1_A2_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8384215, upper bound: 398.8640028
NS_B1_A2_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8384215, upper bound: 398.8640084
NS_B1_A2_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8499079, upper bound: 398.8862838
NS_B1_A2_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8501177, upper bound: 398.8805710
NS_B1_A2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8597428, upper bound: 398.9091543
NS_B1_A2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8635669, upper bound: 398.9105426
NS_B1_A2_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8643831, upper bound: 398.9105097
NS_B1_A2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8643525, upper bound: 398.8895449
NS_B1_A2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8629835, upper bound: 398.8864757
NS_B1_A2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8629878, upper bound: 398.8813791
NS_B1_A2_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8521413, upper bound: 398.8640238
NS_B1_A2_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8640036, upper bound: 398.8875607
NS_B1_A2_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8642422, upper bound: 398.9015026
NS_B1_A2_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8643850, upper bound: 398.9049654
NS_B1_A2_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8638558, upper bound: 398.8885386
NS_B1_A2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8644972, upper bound: 398.8883623
NS_B1_A2_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8600052, upper bound: 398.8602228
NS_B1_A2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8600052, upper bound: 398.8690146
NS_B1_A2_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8642432, upper bound: 398.8864345
NS_B1_A2_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8643984, upper bound: 398.8806098
NS_B1_A2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8770949, upper bound: 398.9090225
NS_B1_A2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8770467, upper bound: 398.8881524
NS_B1_A2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8786176, upper bound: 398.9105942
NS_B1_A2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8785603, upper bound: 398.8896031
NS_B1_A2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8799820, upper bound: 398.8703388
NS_B1_A2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8810380, upper bound: 398.8881870
NS_B1_A2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8808327, upper bound: 398.8710670
NS_B1_A2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8817347, upper bound: 398.8881808
NS_B2_A1_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.9017437, upper bound: 398.8405622
NS_B2_A1_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.9050041, upper bound: 398.8406865
NS_B2_A1_B1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8884357, upper bound: 398.8495646
NS_B2_A1_B1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8883002, upper bound: 398.8502246
NS_B2_A1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8640028, upper bound: 398.8384215
NS_B2_A1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8640028, upper bound: 398.8384215
NS_B2_A1_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8862838, upper bound: 398.8499079
NS_B2_A1_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8805710, upper bound: 398.8501177
NS_B2_A1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.9091543, upper bound: 398.8597428
NS_B2_A1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.9105426, upper bound: 398.8635669
NS_B2_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.9105097, upper bound: 398.8643831
NS_B2_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8895449, upper bound: 398.8643525
NS_B2_A1_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8864757, upper bound: 398.8629835
NS_B2_A1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8813791, upper bound: 398.8629878
NS_B2_A1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8640238, upper bound: 398.8521413
NS_B2_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8875607, upper bound: 398.8640036
NS_B2_A1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.9015026, upper bound: 398.8642422
NS_B2_A1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.9049654, upper bound: 398.8643850
NS_B2_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8885386, upper bound: 398.8638558
NS_B2_A1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8883623, upper bound: 398.8644972
NS_B2_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8602228, upper bound: 398.8600052
NS_B2_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8602228, upper bound: 398.8600052
NS_B2_A1_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8864345, upper bound: 398.8642432
NS_B2_A1_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8806098, upper bound: 398.8643984
NS_B2_A1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.9090225, upper bound: 398.8770949
NS_B2_A1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8881524, upper bound: 398.8770467
NS_B2_A1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.9105942, upper bound: 398.8786176
NS_B2_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8896031, upper bound: 398.8785603
NS_B2_A1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8703388, upper bound: 398.8799820
NS_B2_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8881871, upper bound: 398.8810380
NS_B2_A1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8710670, upper bound: 398.8808327
NS_B2_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8881808, upper bound: 398.8817347
NS_B2_A2_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8131496, upper bound: 398.8783316
NS_B2_A2_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8131496, upper bound: 398.8805966
NS_B2_A2_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8131496, upper bound: 398.8783316
NS_B2_A2_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8131496, upper bound: 398.8807346
NS_B2_A2_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8602472, upper bound: 398.8789031
NS_B2_A2_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8602472, upper bound: 398.8793485
NS_B2_A2_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8618052, upper bound: 398.8789031
NS_B2_A2_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8618052, upper bound: 398.8808937
NS_B2_A2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8200840, upper bound: 398.8783631
NS_B2_A2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8200840, upper bound: 398.8806619
NS_B2_A2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8200840, upper bound: 398.8783631
NS_B2_A2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8200840, upper bound: 398.8807988
NS_B2_A2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8782432, upper bound: 398.8790301
NS_B2_A2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8782432, upper bound: 398.8790301
NS_B2_A2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8782432, upper bound: 398.8795060
NS_B2_A2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.56
Output dim: 0, lower bound: -398.8782432, upper bound: 398.8810410

## BFS NS instance: NS_B1_A1_A1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -156.1449127, 251.0198059, -166.6065979, 271.2405701, -427.3854980, 417.6263428
1: -171.3356628, 222.9051056, -183.0555725, 240.7987671, -412.1343994, 405.9606018
2: -171.1851196, 226.8253632, -182.8773804, 245.2355652, -416.4206848, 409.7027283
3: -200.8874664, 256.8955994, -215.4282684, 277.4872437, -478.3746338, 472.3238525
4: -173.1624146, 260.9781189, -185.4318542, 281.6902771, -454.8526306, 446.4099731

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_A1_A1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_A1_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8666103, upper bound: 398.8878821
time: 0.89 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8635984, upper bound: 398.8876755
time: 0.84 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -156.1449127, 251.0198059, -168.3358459, 275.4323730, -431.5772705, 419.3555603
1: -171.3356628, 222.9051056, -184.8941803, 244.3740692, -415.7096252, 407.7992554
2: -171.1851196, 226.8253632, -184.8663177, 248.7947693, -419.9798279, 411.6916809
3: -200.8874664, 256.8955994, -217.7781677, 281.6236877, -482.5110779, 474.6737671
4: -173.1624146, 260.9781189, -187.5625153, 285.7385559, -458.9009399, 448.5405884

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_A1_A1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_A1_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8666103, upper bound: 398.8878821
time: 1.21 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8635984, upper bound: 398.8876755
time: 0.82 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -158.3645020, 255.9281158, -166.6065979, 271.2405701, -429.6050415, 422.5347290
1: -173.7190857, 227.0350189, -183.0555725, 240.7987671, -414.5178223, 410.0905457
2: -173.6867828, 230.9221497, -182.8773804, 245.2355652, -418.9223633, 413.7994690
3: -203.8833466, 261.6620178, -215.4282684, 277.4872437, -481.3705750, 477.0902710
4: -175.7831421, 265.6315613, -185.4318542, 281.6902771, -457.4733887, 451.0634155

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_A1_A1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_A1_B1_A2_B1_A1

### Relational analysis result of NS_B1_A1_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8543141, upper bound: 398.8886480
time: 0.67 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_A2_B1_A2

### Relational analysis result of NS_B1_A1_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8588994, upper bound: 398.8886419
time: 1.06 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -158.3645020, 255.9281158, -168.3358459, 275.4323730, -433.7968140, 424.2639771
1: -173.7190857, 227.0350189, -184.8941803, 244.3740692, -418.0930481, 411.9291992
2: -173.6867828, 230.9221497, -184.8663177, 248.7947693, -422.4815063, 415.7884521
3: -203.8833466, 261.6620178, -217.7781677, 281.6236877, -485.5070190, 479.4401855
4: -175.7831421, 265.6315613, -187.5625153, 285.7385559, -461.5216675, 453.1940918

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_A1_A1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_A1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A1_A1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8543141, upper bound: 398.8886480
time: 0.81 seconds

## Relational analysis of NS_B1_A1_A1_A1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A1_A1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8588994, upper bound: 398.8886419
time: 1.11 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -156.1449127, 251.0198059, -159.6785431, 263.7409363, -419.8858643, 410.6982727
1: -171.3356628, 222.9051056, -175.7512207, 233.9708557, -405.3064270, 398.6562805
2: -171.1851196, 226.8253632, -175.6281433, 238.2758789, -409.4609375, 402.4534912
3: -200.8874664, 256.8955994, -207.7480316, 269.5384827, -470.4258728, 464.6436157
4: -173.1624146, 260.9781189, -178.7513733, 273.4935913, -446.6560059, 439.7294922

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8646872, upper bound: 398.8877443
time: 1.31 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8637088, upper bound: 398.8875471
time: 0.96 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -156.1449127, 251.0198059, -162.6402283, 270.3965454, -426.5414429, 413.6599121
1: -171.3356628, 222.9051056, -179.0090942, 239.8341370, -411.1697083, 401.9141541
2: -171.1851196, 226.8253632, -179.0359497, 244.1249542, -415.3100586, 405.8612976
3: -200.8874664, 256.8955994, -211.9325409, 276.3398438, -477.2272034, 468.8281250
4: -173.1624146, 260.9781189, -182.4304657, 280.1494141, -453.3117981, 443.4085693

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B2_A1

### Relational analysis result of NS_B1_A1_A1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8773396, upper bound: 398.8894461
time: 1.15 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B2_A1

### Relational analysis result of NS_B1_A1_A1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8646872, upper bound: 398.8877443
time: 1.11 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_A1_B2_A2

### Relational analysis result of NS_B1_A1_A1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8637088, upper bound: 398.8875471
time: 0.98 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -158.3645020, 255.9281158, -159.6618042, 263.7184143, -422.0829163, 415.5899048
1: -173.7190857, 227.0350189, -175.7336426, 233.9511871, -407.6702576, 402.7686768
2: -173.6867828, 230.9221497, -175.6102142, 238.2553558, -411.9421387, 406.5323486
3: -203.8833466, 261.6620178, -207.7288513, 269.5157166, -473.3990479, 469.3908691
4: -175.7831421, 265.6315613, -178.7344055, 273.4699097, -449.2529907, 444.3659668

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B1_B1

### Relational analysis result of NS_B1_A1_A1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8600226, upper bound: 398.8874424
time: 0.90 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B1_B2

### Relational analysis result of NS_B1_A1_A1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8599626, upper bound: 398.8886469
time: 1.49 seconds

## BFS NS instance: NS_B1_A1_A1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -158.3645020, 255.9281158, -162.6402283, 270.3965454, -428.7610168, 418.5683594
1: -173.7190857, 227.0350189, -179.0090942, 239.8341370, -413.5531311, 406.0441284
2: -173.6867828, 230.9221497, -179.0359497, 244.1249542, -417.8117371, 409.9580994
3: -203.8833466, 261.6620178, -211.9325409, 276.3398438, -480.2231750, 473.5945435
4: -175.7831421, 265.6315613, -182.4304657, 280.1494141, -455.9325256, 448.0620117

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B2_B1

### Relational analysis result of NS_B1_A1_A1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8600226, upper bound: 398.8874424
time: 0.78 seconds

## Relational analysis of NS_B1_A1_A1_A1_B2_A2_B2_B2

### Relational analysis result of NS_B1_A1_A1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8599626, upper bound: 398.8886469
time: 1.26 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -148.0388947, 241.5237885, -166.6065979, 271.2405701, -419.2794495, 408.1303711
1: -162.6600037, 214.2815247, -183.0555725, 240.7987671, -403.4587708, 397.3370056
2: -162.6134644, 217.9182739, -182.8773804, 245.2355652, -407.8490295, 400.7955933
3: -191.5447998, 246.8700562, -215.4282684, 277.4872437, -469.0320435, 462.2983093
4: -165.1260376, 250.5240936, -185.4318542, 281.6902771, -446.8162537, 435.9559021

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8543259, upper bound: 398.8939104
time: 0.89 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A1_A2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8543259, upper bound: 398.8939104
time: 1.01 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -151.3585052, 247.5527039, -166.6065979, 271.2405701, -422.5990601, 414.1593018
1: -166.3431549, 219.5398407, -183.0555725, 240.7987671, -407.1418762, 402.5953064
2: -166.3361969, 223.2579956, -182.8773804, 245.2355652, -411.5717773, 406.1353455
3: -196.0121155, 252.9013672, -215.4282684, 277.4872437, -473.4993286, 468.3295898
4: -168.9264526, 256.7081299, -185.4318542, 281.6902771, -450.6166382, 442.1399536

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A2_A1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8616591, upper bound: 398.8938938
time: 0.96 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A2_A2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8616591, upper bound: 398.8938938
time: 0.97 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -145.1505280, 237.8262787, -166.3829346, 271.9928894, -417.1434326, 404.2092285
1: -159.5099487, 210.8143921, -182.7352295, 241.3312531, -400.8411560, 393.5496216
2: -159.5144653, 214.3631287, -182.7033691, 245.7251892, -405.2395630, 397.0664978
3: -187.9609528, 242.8783875, -215.2109833, 278.1181030, -466.0790405, 458.0893250
4: -162.0800781, 246.4416351, -185.3429413, 282.2292480, -444.3093262, 431.7845764

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8707659, upper bound: 398.8766933
time: 0.90 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A1_B2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8707659, upper bound: 398.8766933
time: 0.90 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -152.3439636, 249.1921539, -166.3493042, 271.7627563, -424.1066895, 415.5414429
1: -167.4247131, 220.9404755, -182.6977234, 241.1196899, -408.5444031, 403.6381836
2: -167.3980713, 224.6333771, -182.6525116, 245.5166016, -412.9146118, 407.2858582
3: -197.2675476, 254.5650177, -215.1575928, 277.9074097, -475.1748962, 469.7225952
4: -169.9976349, 258.2374573, -185.2745209, 282.0344238, -452.0320435, 443.5119019

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A2_A1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8464254, upper bound: 398.8735976
time: 0.96 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A2_A2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8574355, upper bound: 398.8736411
time: 1.02 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -148.8428497, 242.8462067, -159.6785431, 263.7409363, -412.5837402, 402.5247498
1: -163.5882111, 215.5297394, -175.7512207, 233.9708557, -397.5589905, 391.2809143
2: -163.5106354, 219.1772308, -175.6281433, 238.2758789, -401.7864990, 394.8053284
3: -192.6372375, 248.3060455, -207.7480316, 269.5384827, -462.1757202, 456.0540771
4: -166.1027527, 251.9715881, -178.7513733, 273.4935913, -439.5963440, 430.7229614

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A1_A2_B2_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8900302, upper bound: 398.8890915
time: 1.04 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8936307, upper bound: 398.8911555
time: 1.14 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -148.8428497, 242.8462067, -162.6402283, 270.3965454, -419.2393494, 405.4863892
1: -163.5882111, 215.5297394, -179.0090942, 239.8341370, -403.4222717, 394.5387878
2: -163.5106354, 219.1772308, -179.0359497, 244.1249542, -407.6355896, 398.2131348
3: -192.6372375, 248.3060455, -211.9325409, 276.3398438, -468.9770508, 460.2385559
4: -166.1027527, 251.9715881, -182.4304657, 280.1494141, -446.2521362, 434.4020386

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_A1_A1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A1_A2_B2_A1_B2_A1

### Relational analysis result of NS_B1_A1_A1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8900302, upper bound: 398.8890915
time: 0.93 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2_A1_B2_A2

### Relational analysis result of NS_B1_A1_A1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -398.8936307, upper bound: 398.8911555
time: 0.98 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.21 + 417.21 = 420.42 seconds
