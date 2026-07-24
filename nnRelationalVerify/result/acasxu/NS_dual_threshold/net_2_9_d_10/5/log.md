## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 5693.26040512119


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3552.1267090, 2750.7805176, -3552.1267090, 2750.7805176, -6302.9072266, 6302.9072266)
1: (-294.4743042, 208.4085846, -294.4743042, 208.4085846, -502.8828430, 502.8828430)
2: (-202.6398773, 349.5717773, -202.6398773, 349.5717773, -552.2115479, 552.2115479)
3: (-246.1018677, 512.2490845, -246.1018677, 512.2490845, -758.3508301, 758.3508301)
4: (-197.5094757, 359.1516113, -197.5094757, 359.1516113, -556.6610718, 556.6610718)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.99 + 1.91 = 4.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -5693.8297878, upper bound: 5693.8297881

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297681, upper bound: 5693.8297547
time: 0.61 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297881, upper bound: 5693.8297881
time: 0.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.44 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 0, lower bound: -5693.8297681, upper bound: 5693.8297547
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.44
Output dim: 0, lower bound: -5693.8297881, upper bound: 5693.8297881

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3634.5844727, 2774.4235840, -3542.8081055, 2742.8576660, -6377.4418945, 6317.2314453
1: -298.3411255, 213.9223328, -293.6480103, 207.8511505, -506.1922302, 507.5703430
2: -209.4519196, 355.2555542, -202.1083679, 348.6013489, -558.0532837, 557.3638916
3: -253.6786194, 522.1437378, -245.4545135, 510.8419495, -764.5205078, 767.5982666
4: -203.2445679, 364.4107666, -196.9744873, 358.1553650, -561.3999023, 561.3852539

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8296726, upper bound: 5693.8297508
time: 0.69 seconds

## Relational analysis of NS_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297406, upper bound: 5693.8297480
time: 0.64 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297406, upper bound: 5693.8297393
time: 0.79 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3548.4016113, 2748.1281738, -3552.1267090, 2750.7805176, -6299.1821289, 6300.2548828
1: -294.1842651, 208.1901855, -294.4743042, 208.4085846, -502.5928040, 502.6644592
2: -202.4424744, 349.2319031, -202.6398773, 349.5717773, -552.0141602, 551.8715820
3: -245.8611450, 511.7356873, -246.1018677, 512.2490845, -758.1101685, 757.8375244
4: -197.3195038, 358.8009338, -197.5094757, 359.1516113, -556.4711304, 556.3104248

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297880, upper bound: 5693.8297880
time: 0.60 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297881, upper bound: 5693.8297881
time: 0.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.33 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 4.33
Output dim: 0, lower bound: -5693.8297406, upper bound: 5693.8297480
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 4.33
Output dim: 0, lower bound: -5693.8297406, upper bound: 5693.8297393
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.33
Output dim: 0, lower bound: -5693.8297880, upper bound: 5693.8297880
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.33
Output dim: 0, lower bound: -5693.8297881, upper bound: 5693.8297881

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -3572.2783203, 2728.6286621, -3542.8081055, 2742.8576660, -6315.1357422, 6271.4365234
1: -293.3261719, 210.1761932, -293.6480103, 207.8511505, -501.1772766, 503.8242188
2: -205.7708740, 349.1928406, -202.1083679, 348.6013489, -554.3721313, 551.3012085
3: -249.2654419, 513.1068726, -245.4545135, 510.8419495, -760.1074219, 758.5614014
4: -199.7129517, 358.1905212, -196.9744873, 358.1553650, -557.8682861, 555.1650391

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297405, upper bound: 5693.8297393
time: 0.70 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297406, upper bound: 5693.8297392
time: 0.65 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -3977.6835938, 3048.8144531, -3466.6181641, 2686.2182617, -6663.9013672, 6515.4326172
1: -327.2920532, 234.8981934, -287.5036926, 203.3076019, -530.5996704, 522.4017334
2: -228.9180450, 389.9218140, -197.3612671, 341.1400146, -570.0580444, 587.2830811
3: -277.5253296, 573.5651245, -239.7509460, 499.8850098, -777.4102783, 813.3160400
4: -222.8388214, 400.6781921, -192.4243774, 350.5803223, -573.4191284, 593.1024170

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297405, upper bound: 5693.8297392
time: 0.53 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297405, upper bound: 5693.8297392
time: 0.53 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -3254.8784180, 2551.9035645, -3391.8928223, 2644.3701172, -5899.2470703, 5943.7954102
1: -272.3960571, 191.0052490, -282.6142273, 199.0559387, -471.4519958, 473.6194458
2: -185.6618042, 322.8197327, -193.4045715, 335.2016602, -520.8633423, 516.2243042
3: -225.7985840, 472.2024536, -235.0782471, 490.6989746, -716.4975586, 707.2807007
4: -181.3931732, 331.5055237, -188.7209167, 344.2978516, -525.6908569, 520.2264404

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297830, upper bound: 5693.8297852
time: 0.63 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297851, upper bound: 5693.8297852
time: 0.65 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -3705.2126465, 2910.8188477, -3474.2268066, 2697.4882812, -6402.7011719, 6385.0458984
1: -310.7076721, 218.3763275, -288.5811157, 203.8171387, -514.5247803, 506.9574585
2: -215.6065979, 369.5586243, -198.0830231, 342.5096130, -558.1160889, 567.6416626
3: -261.8283691, 538.5621948, -240.6807404, 501.6675720, -763.4959106, 779.2429199
4: -210.2556763, 378.9247131, -193.1802673, 351.8709412, -562.1265869, 572.1049194

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297881, upper bound: 5693.8297881
time: 0.61 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297881, upper bound: 5693.8297881
time: 0.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.20 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -5693.8297405, upper bound: 5693.8297393
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -5693.8297406, upper bound: 5693.8297392
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -5693.8297405, upper bound: 5693.8297392
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -5693.8297405, upper bound: 5693.8297392
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -5693.8297830, upper bound: 5693.8297852
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -5693.8297851, upper bound: 5693.8297852
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -5693.8297881, upper bound: 5693.8297881
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.20
Output dim: 0, lower bound: -5693.8297881, upper bound: 5693.8297881

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -3572.2783203, 2728.6286621, -3482.5842285, 2696.9919434, -6269.2705078, 6211.2128906
1: -293.3261719, 210.1761932, -288.7021790, 204.2304535, -497.5565796, 498.8783264
2: -205.7708740, 349.1928406, -198.4234009, 342.6437988, -548.4146729, 547.6162109
3: -249.2654419, 513.1068726, -241.0176239, 502.1546631, -751.4201050, 754.1245117
4: -199.7129517, 358.1905212, -193.4523773, 352.0478821, -551.7608643, 551.6428833

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297392, upper bound: 5693.8297481
time: 0.57 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297392, upper bound: 5693.8297481
time: 0.55 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -3572.2783203, 2728.6286621, -3952.8715820, 3067.1196289, -6639.3979492, 6681.5000000
1: -293.3261719, 210.1761932, -328.0470581, 232.9381256, -526.2642822, 538.2232666
2: -205.7708740, 349.1928406, -225.4100037, 389.2892151, -595.0600586, 574.6027222
3: -249.2654419, 513.1068726, -273.8244934, 572.1399536, -821.4053955, 786.9313965
4: -199.7129517, 358.1905212, -220.1890869, 400.4930115, -600.2059326, 578.3795776

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297392, upper bound: 5693.8297481
time: 0.53 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297392, upper bound: 5693.8297480
time: 0.68 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -3977.6835938, 3048.8144531, -3480.9189453, 2695.7678223, -6673.4511719, 6529.7324219
1: -327.2920532, 234.8981934, -288.5690002, 204.1304016, -531.4224854, 523.4671021
2: -228.9180450, 389.9218140, -198.3194275, 342.4779358, -571.3958740, 588.2412109
3: -277.5253296, 573.5651245, -240.8936157, 501.9122314, -779.4373779, 814.4586792
4: -222.8388214, 400.6781921, -193.3526917, 351.8796692, -574.7185059, 594.0307617

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8296571, upper bound: 5693.8297244
time: 0.67 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297393, upper bound: 5693.8297383
time: 0.59 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3977.6835938, 3048.8144531, -3952.8715820, 3067.1196289, -7044.8017578, 7001.6860352
1: -327.2920532, 234.8981934, -328.0470581, 232.9381256, -560.2301636, 562.9451294
2: -228.9180450, 389.9218140, -225.4100037, 389.2892151, -618.2072144, 615.3316650
3: -277.5253296, 573.5651245, -273.8244934, 572.1399536, -849.6652832, 847.3895874
4: -222.8388214, 400.6781921, -220.1890869, 400.4930115, -623.3317871, 620.8671265

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297366, upper bound: 5693.8297308
time: 0.65 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297365, upper bound: 5693.8297368
time: 0.70 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -3220.5336914, 2526.7636719, -3318.6660156, 2590.5610352, -5811.0947266, 5845.4291992
1: -269.6999817, 188.9422760, -276.8489685, 194.6352539, -464.3352356, 465.7912598
2: -183.6357727, 319.5047607, -189.0698700, 328.1137085, -511.7493896, 508.5746460
3: -223.3802948, 467.4065247, -229.8885040, 480.4348145, -703.8151245, 697.2950439
4: -179.4936981, 328.1122437, -184.6378021, 337.0209351, -516.5145874, 512.7499390

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297692, upper bound: 5693.8297749
time: 0.59 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297714, upper bound: 5693.8297750
time: 0.64 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -3203.6765137, 2515.5529785, -3335.0253906, 2610.3354492, -5814.0112305, 5850.5776367
1: -268.4539795, 187.9934998, -278.8074646, 195.7806091, -464.2345886, 466.8009644
2: -182.7095184, 318.0401306, -190.0039520, 330.3663330, -513.0758057, 508.0440369
3: -222.2631683, 465.2069702, -231.0709229, 483.3266296, -705.5896606, 696.2778931
4: -178.6062775, 326.5918579, -185.5935974, 339.2518311, -517.8580933, 512.1854248

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297733, upper bound: 5693.8297751
time: 0.69 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297749, upper bound: 5693.8297752
time: 0.76 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -3705.2126465, 2910.8188477, -3258.4960938, 2554.5190430, -6259.7314453, 6169.3149414
1: -310.7076721, 218.3763275, -272.6817322, 191.2189178, -501.9265747, 491.0580444
2: -215.6065979, 369.5586243, -185.8512726, 323.1545105, -538.7610474, 555.4098511
3: -261.8283691, 538.5621948, -226.0303497, 472.7031250, -734.5314331, 764.5925293
4: -210.2556763, 378.9247131, -181.5751495, 331.8414612, -542.0971680, 560.4998779

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297747, upper bound: 5693.8297790
time: 0.87 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297788, upper bound: 5693.8297785
time: 0.74 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -3705.2126465, 2910.8188477, -3709.0078125, 2913.5207520, -6618.7333984, 6619.8266602
1: -310.7076721, 218.3763275, -311.0048828, 218.6035919, -529.3112793, 529.3811646
2: -215.6065979, 369.5586243, -215.8076630, 369.9015198, -585.5081177, 585.3662720
3: -261.8283691, 538.5621948, -262.0737610, 539.0894165, -800.9177856, 800.6359863
4: -210.2556763, 378.9247131, -210.4501495, 379.2772217, -589.5328369, 589.3748169

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297859, upper bound: 5693.8297872
time: 0.55 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297860, upper bound: 5693.8297864
time: 0.59 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.21 seconds
NS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 0, lower bound: -5693.8297392, upper bound: 5693.8297481
NS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 0, lower bound: -5693.8297392, upper bound: 5693.8297481
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 0, lower bound: -5693.8297392, upper bound: 5693.8297481
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 0, lower bound: -5693.8297392, upper bound: 5693.8297480
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 0, lower bound: -5693.8296571, upper bound: 5693.8297244
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 0, lower bound: -5693.8297393, upper bound: 5693.8297383
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 0, lower bound: -5693.8297366, upper bound: 5693.8297308
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 0, lower bound: -5693.8297365, upper bound: 5693.8297368
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 0, lower bound: -5693.8297692, upper bound: 5693.8297749
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 0, lower bound: -5693.8297714, upper bound: 5693.8297750
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 0, lower bound: -5693.8297733, upper bound: 5693.8297751
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 0, lower bound: -5693.8297749, upper bound: 5693.8297752
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 0, lower bound: -5693.8297747, upper bound: 5693.8297790
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 0, lower bound: -5693.8297788, upper bound: 5693.8297785
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 0, lower bound: -5693.8297859, upper bound: 5693.8297872
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.21
Output dim: 0, lower bound: -5693.8297860, upper bound: 5693.8297864

## BFS NS instance: NS_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -3572.2783203, 2728.6286621, -3572.2783203, 2728.6286621, -6300.9072266, 6300.9072266
1: -293.3261719, 210.1761932, -293.3261719, 210.1761932, -503.5023499, 503.5023499
2: -205.7708740, 349.1928406, -205.7708740, 349.1928406, -554.9636841, 554.9636841
3: -249.2654419, 513.1068726, -249.2654419, 513.1068726, -762.3723145, 762.3723145
4: -199.7129517, 358.1905212, -199.7129517, 358.1905212, -557.9034424, 557.9034424

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

## BFS NS instance: NS_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -3572.2783203, 2728.6286621, -3488.4379883, 2702.3769531, -6274.6552734, 6217.0664062
1: -293.3261719, 210.1761932, -289.2507019, 204.5848083, -497.9109192, 499.4268799
2: -205.7708740, 349.1928406, -198.7722778, 343.2878113, -549.0587158, 547.9650879
3: -249.2654419, 513.1068726, -241.4403687, 503.0873108, -752.3527832, 754.5472412
4: -199.7129517, 358.1905212, -193.8111725, 352.7093201, -552.4222412, 552.0017090

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 35

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -3572.2783203, 2728.6286621, -3892.1525879, 2980.8791504, -6553.1572266, 6620.7812500
1: -293.3261719, 210.1761932, -320.0213013, 229.5803070, -522.9064331, 530.1975098
2: -205.7708740, 349.1928406, -223.4593048, 380.9909363, -586.7617798, 572.6521606
3: -249.2654419, 513.1068726, -270.9670105, 560.4573975, -809.7228394, 784.0738525
4: -199.7129517, 358.1905212, -217.3618774, 391.6401062, -591.3529663, 575.5523682

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 24

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -3572.2783203, 2728.6286621, -3959.4499512, 3073.0358887, -6645.3139648, 6688.0786133
1: -293.3261719, 210.1761932, -328.6610413, 233.3391113, -526.6652832, 538.8372192
2: -205.7708740, 349.1928406, -225.7780914, 390.0014648, -595.7723389, 574.9708862
3: -249.2654419, 513.1068726, -274.2703247, 573.1726685, -822.4381104, 787.3771362
4: -199.7129517, 358.1905212, -220.5654297, 401.2269287, -600.9398804, 578.7559814

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -3977.6835938, 3048.8144531, -3399.5441895, 2633.0039062, -6610.6865234, 6448.3579102
1: -327.2920532, 234.8981934, -281.8894958, 199.1801758, -526.4722290, 516.7875366
2: -228.9180450, 389.9218140, -193.4111938, 334.4069519, -563.3248901, 583.3330078
3: -277.5253296, 573.5651245, -234.9882660, 490.2301025, -767.7554321, 808.5534058
4: -222.8388214, 400.6781921, -188.7224274, 343.6427002, -566.4815063, 589.4006348

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8296084, upper bound: 5693.8297305
time: 0.53 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8296902, upper bound: 5693.8297311
time: 0.65 seconds

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -3899.9902344, 2988.9799805, -3684.5415039, 2826.1894531, -6726.1787109, 6673.5214844
1: -320.8756104, 230.2120056, -302.9695740, 215.0227051, -535.8982544, 533.1815796
2: -224.0843201, 382.2762146, -208.1882324, 359.3370361, -583.4212036, 590.4644775
3: -271.7688293, 562.3516235, -252.9768372, 528.1027222, -799.8715820, 815.3283691
4: -218.2110138, 392.8867188, -203.0008087, 369.6792603, -587.8902588, 595.8875122

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297363, upper bound: 5693.8297305
time: 0.54 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2

### Relational analysis result of NS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297575, upper bound: 5693.8297363
time: 0.69 seconds

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -3934.8764648, 3019.1643066, -3883.9882812, 3019.6596680, -6954.5361328, 6903.1523438
1: -323.9678345, 232.3616486, -322.7070007, 228.8570709, -552.8248291, 555.0686035
2: -226.3373260, 385.9615173, -221.1973572, 382.9295044, -609.2668457, 607.1588135
3: -274.3649292, 567.6581421, -268.6929626, 562.6245117, -836.9893188, 836.3510742
4: -220.2830048, 396.6012268, -216.0549164, 393.9274902, -614.2104492, 612.6560669

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297362, upper bound: 5693.8297183
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297362, upper bound: 5693.8297309
time: 0.62 seconds

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3932.6416016, 3011.2233887, -3912.7197266, 3027.2236328, -6959.8652344, 6923.9433594
1: -323.3285522, 232.1136017, -323.9695129, 230.2490082, -553.5775757, 556.0831299
2: -226.1192627, 385.3071899, -222.6802979, 384.5722961, -610.6915283, 607.9872437
3: -274.2231445, 566.9456787, -270.5864258, 565.6382446, -839.8613892, 837.5321045
4: -220.1730957, 395.9217224, -217.5697174, 395.6221924, -615.7952881, 613.4914551

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297376, upper bound: 5693.8297183
time: 0.55 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297377, upper bound: 5693.8297368
time: 0.69 seconds

## BFS NS instance: NS_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -3183.5341797, 2499.0415039, -3207.4638672, 2507.9367676, -5691.4707031, 5706.5053711
1: -266.7555542, 186.7019653, -268.0381470, 187.9055634, -454.6611328, 454.7400208
2: -181.3873901, 315.8700256, -182.3226624, 317.2563782, -498.6437378, 498.1926880
3: -220.7001038, 462.1286926, -221.8622131, 464.6018066, -685.3018188, 683.9909058
4: -177.3754120, 324.3885498, -178.2803650, 325.9203796, -503.2957764, 502.6689148

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_A1_B1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297467, upper bound: 5693.8297696
time: 0.72 seconds

## Relational analysis of NS_A2_A1_B1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297671, upper bound: 5693.8297728
time: 0.69 seconds

## BFS NS instance: NS_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -3168.8627930, 2489.0734863, -3296.6572266, 2577.4282227, -5746.2910156, 5785.7299805
1: -265.6513062, 185.8263550, -275.3847656, 193.0762482, -458.7275391, 461.2111206
2: -180.5597687, 314.5932617, -187.5606384, 326.2614136, -506.8211060, 502.1539001
3: -219.7264709, 460.1086121, -228.2343597, 477.5340271, -697.2604370, 688.3429565
4: -176.5665131, 323.0501099, -183.4118500, 335.1685486, -511.7350464, 506.4619751

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_A1_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297669, upper bound: 5693.8297625
time: 0.57 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297684, upper bound: 5693.8297724
time: 0.69 seconds

## BFS NS instance: NS_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -3166.5739746, 2487.6130371, -3226.4780273, 2528.9558105, -5695.5297852, 5714.0903320
1: -265.4946899, 185.7454376, -270.1625671, 189.1931763, -454.6878357, 455.9080200
2: -180.4491882, 314.3843384, -183.3457489, 319.7029724, -500.1521301, 497.7300415
3: -219.5620880, 459.9030762, -223.1637421, 467.8512573, -687.4133301, 683.0668335
4: -176.4750214, 322.8525696, -179.3373260, 328.3417664, -504.8167725, 502.1898804

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297496, upper bound: 5693.8297625
time: 0.70 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297730
time: 0.56 seconds

## BFS NS instance: NS_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -3152.1591797, 2478.1765137, -3316.4689941, 2599.4162598, -5751.5751953, 5794.6455078
1: -264.4385986, 184.8881531, -277.6361084, 194.5149841, -458.9535522, 462.5242615
2: -179.6528625, 313.1706848, -188.6209106, 328.8367920, -508.4896545, 501.7915955
3: -218.6321564, 457.9486084, -229.5708313, 481.0058899, -699.6380005, 687.5194092
4: -175.7009888, 321.5695190, -184.5027161, 337.6728516, -513.3737793, 506.0722046

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297674, upper bound: 5693.8297623
time: 0.93 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297725, upper bound: 5693.8297728
time: 0.80 seconds

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -3662.5832520, 2878.0954590, -3147.3930664, 2471.7109375, -6134.2939453, 6025.4882812
1: -307.2225037, 215.7569580, -263.8736267, 184.4954224, -491.7179260, 479.6305542
2: -212.9251556, 365.3077698, -179.1030579, 312.2913513, -525.2163696, 544.4107666
3: -258.5960083, 532.3982544, -217.9930573, 456.9090271, -715.5050049, 750.3911743
4: -207.6816864, 374.5820923, -175.2185516, 320.7391968, -528.4208984, 549.8006592

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297705, upper bound: 5693.8297721
time: 0.71 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297703, upper bound: 5693.8297750
time: 0.79 seconds

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -3644.0527344, 2867.3706055, -3225.6752930, 2533.9885254, -6178.0410156, 6093.0444336
1: -305.9739380, 214.6928101, -270.4266663, 189.0621490, -495.0360718, 485.1194153
2: -212.0109100, 363.8758240, -183.7265472, 320.3342590, -532.3451538, 547.6023560
3: -257.5277710, 530.0294800, -223.6351318, 468.4119873, -725.9397583, 753.6646118
4: -206.8103485, 373.0671082, -179.7709198, 328.9851685, -535.7954102, 552.8380127

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297750, upper bound: 5693.8297713
time: 0.65 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297751, upper bound: 5693.8297744
time: 0.67 seconds

## BFS NS instance: NS_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3581.0471191, 2821.2988281, -3633.7485352, 2859.2509766, -6440.2978516, 6455.0473633
1: -300.8927612, 210.8540802, -305.0578613, 214.0406036, -514.9333496, 515.9119263
2: -208.0160370, 357.9273376, -211.2150116, 362.8504639, -570.8663330, 569.1423340
3: -252.7596741, 521.2930298, -256.5817566, 528.6220093, -781.3816528, 777.8747559
4: -203.0079193, 366.9593506, -206.0634766, 372.0222473, -575.0301514, 573.0227661

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297864, upper bound: 5693.8297864
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297864, upper bound: 5693.8297864
time: 0.61 seconds

## BFS NS instance: NS_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3808.3254395, 3001.3354492, -3644.3522949, 2864.6391602, -6672.9648438, 6645.6875000
1: -319.8797607, 225.0525970, -305.7066040, 214.7049408, -534.5847168, 530.7590942
2: -221.8701630, 380.8742065, -211.7078400, 363.6033325, -585.4735107, 592.5820312
3: -269.3688965, 554.8281250, -257.1600952, 529.8810425, -799.2499390, 811.9882202
4: -216.5942535, 390.5656738, -206.5190887, 372.8416138, -589.4357910, 597.0847168

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297864, upper bound: 5693.8297864
time: 0.78 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297864, upper bound: 5693.8297864
time: 0.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.63 seconds
NS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8296084, upper bound: 5693.8297305
NS_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8296902, upper bound: 5693.8297311
NS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297363, upper bound: 5693.8297305
NS_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297575, upper bound: 5693.8297363
NS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297362, upper bound: 5693.8297183
NS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297362, upper bound: 5693.8297309
NS_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297376, upper bound: 5693.8297183
NS_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297377, upper bound: 5693.8297368
NS_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297467, upper bound: 5693.8297696
NS_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297671, upper bound: 5693.8297728
NS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297669, upper bound: 5693.8297625
NS_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297684, upper bound: 5693.8297724
NS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297496, upper bound: 5693.8297625
NS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297722, upper bound: 5693.8297730
NS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297674, upper bound: 5693.8297623
NS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297725, upper bound: 5693.8297728
NS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297705, upper bound: 5693.8297721
NS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297703, upper bound: 5693.8297750
NS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297750, upper bound: 5693.8297713
NS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297751, upper bound: 5693.8297744
NS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297864, upper bound: 5693.8297864
NS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297864, upper bound: 5693.8297864
NS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297864, upper bound: 5693.8297864
NS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.63
Output dim: 0, lower bound: -5693.8297864, upper bound: 5693.8297864

## BFS NS instance: NS_A1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -3933.0004883, 3015.4887695, -3285.6853027, 2547.5961914, -6480.5961914, 6301.1738281
1: -323.7180176, 232.1701965, -272.7968140, 192.2500000, -515.9679565, 504.9670105
2: -226.2341156, 385.5911255, -186.4666138, 323.1998291, -549.4338989, 572.0577393
3: -274.2871704, 567.1953735, -226.7087250, 473.9505615, -748.2376099, 793.9041138
4: -220.2341919, 396.2639465, -182.1629333, 332.1903992, -552.4245605, 578.4268799

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B1_B1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8296056, upper bound: 5693.8297297
time: 0.60 seconds

## Relational analysis of NS_A1_A2_B1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -3925.4436035, 3012.7419434, -3377.7617188, 2619.5329590, -6544.9760742, 6390.5034180
1: -323.3258057, 231.7948761, -280.4077148, 197.6360779, -520.9619141, 512.2025146
2: -225.9506073, 385.1629333, -191.8312988, 332.5234070, -558.4738770, 576.9942627
3: -273.9674072, 566.3255615, -233.2580261, 487.3833313, -761.3505859, 799.5836182
4: -219.9580994, 395.7851257, -187.4273682, 341.7766724, -561.7346191, 583.2124634

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B1_B1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8296761, upper bound: 5693.8297308
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -3857.6020508, 2959.4743652, -3617.2749023, 2778.7949219, -6636.3955078, 6576.7490234
1: -317.5697937, 227.6791229, -297.7424622, 211.0207062, -528.5905151, 525.4215088
2: -221.5184631, 378.3332214, -203.9456329, 353.1415100, -574.6599731, 582.2788086
3: -268.6268311, 556.4722900, -247.9740753, 518.9578247, -787.5846558, 804.4462280
4: -215.6645813, 388.8287964, -199.0031128, 363.3136902, -578.9781494, 587.8319092

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B2_B1_A1

### Relational analysis result of NS_A1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297352, upper bound: 5693.8297180
time: 0.61 seconds

## Relational analysis of NS_A1_A2_B1_B2_B1_A2

### Relational analysis result of NS_A1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297352, upper bound: 5693.8297304
time: 0.57 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -3855.8041992, 2952.0585938, -3648.2145996, 2785.5993652, -6641.4023438, 6600.2729492
1: -316.9814758, 227.4855957, -298.9281616, 212.3625183, -529.3439941, 526.4135742
2: -221.3381805, 377.7461243, -205.4341431, 354.4961548, -575.8342896, 583.1802979
3: -268.5187683, 555.8449097, -249.6739197, 521.7534180, -790.2720947, 805.5187988
4: -215.5838776, 388.2064819, -200.3029327, 364.7412109, -580.3250732, 588.5093994

Time for backsubstitution: 2.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_B2_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297414, upper bound: 5693.8297180
time: 0.70 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297412, upper bound: 5693.8297363
time: 0.60 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -3906.3347168, 2999.4653320, -3883.9882812, 3019.6596680, -6925.9941406, 6883.4536133
1: -321.7507019, 230.6793213, -322.7070007, 228.8570709, -550.6077881, 553.3861084
2: -224.6058655, 383.3260193, -221.1973572, 382.9295044, -607.5354004, 604.5233765
3: -272.2572937, 563.7211914, -268.6929626, 562.6245117, -834.8817749, 832.4141846
4: -218.5975037, 393.8852844, -216.0549164, 393.9274902, -612.5249634, 609.9401855

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_B2_B1_A1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297188, upper bound: 5693.8297165
time: 0.62 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B2_B1_A1_B1

### Relational analysis result of NS_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297183, upper bound: 5693.8297183
time: 0.56 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_B2

### Relational analysis result of NS_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297183, upper bound: 5693.8297183
time: 0.51 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -3933.7758789, 3007.9921875, -3883.9882812, 3019.6596680, -6953.4355469, 6891.9799805
1: -323.0980530, 232.0151062, -322.7070007, 228.8570709, -551.9550781, 554.7219238
2: -226.0618286, 385.1126099, -221.1973572, 382.9295044, -608.9912720, 606.3099365
3: -274.1437988, 566.8295898, -268.6929626, 562.6245117, -836.7681885, 835.5225830
4: -220.1205597, 395.7281799, -216.0549164, 393.9274902, -614.0480347, 611.7828979

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_B2_B1_A2_A1

### Relational analysis result of NS_A1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297190, upper bound: 5693.8297167
time: 0.75 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297183, upper bound: 5693.8297183
time: 0.55 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297179, upper bound: 5693.8297309
time: 0.57 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -3906.3347168, 2999.4653320, -3912.7197266, 3027.2236328, -6933.5571289, 6912.1850586
1: -321.7507019, 230.6793213, -323.9695129, 230.2490082, -551.9996948, 554.6487427
2: -224.6058655, 383.3260193, -222.6802979, 384.5722961, -609.1781616, 606.0061035
3: -272.2572937, 563.7211914, -270.5864258, 565.6382446, -837.8955078, 834.3075562
4: -218.5975037, 393.8852844, -217.5697174, 395.6221924, -614.2196655, 611.4550171

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B2_B2_A1_B1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297183, upper bound: 5693.8297183
time: 0.67 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297183, upper bound: 5693.8297183
time: 0.54 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -3933.7758789, 3007.9921875, -3912.7197266, 3027.2236328, -6960.9985352, 6920.7119141
1: -323.0980530, 232.0151062, -323.9695129, 230.2490082, -553.3469849, 555.9845581
2: -226.0618286, 385.1126099, -222.6802979, 384.5722961, -610.6340942, 607.7927246
3: -274.1437988, 566.8295898, -270.5864258, 565.6382446, -839.7820435, 837.4160156
4: -220.1205597, 395.7281799, -217.5697174, 395.6221924, -615.7427368, 613.2977295

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297183, upper bound: 5693.8297367
time: 0.57 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297183, upper bound: 5693.8297368
time: 0.56 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -3108.9841309, 2442.5241699, -3096.6042480, 2423.7744141, -5532.7578125, 5539.1269531
1: -260.7269287, 182.2050171, -259.0422058, 181.2160034, -441.9429321, 441.2472229
2: -176.7001648, 308.5474854, -175.3394623, 306.3741455, -483.0743103, 483.8869324
3: -215.1439209, 451.5631104, -213.5798798, 448.8952637, -664.0391846, 665.1429443
4: -172.9491425, 316.9064026, -171.6853943, 314.8195496, -487.7686768, 488.5917969

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8296519, upper bound: 5693.8297541
time: 0.57 seconds

## Relational analysis of NS_A2_A1_B1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A1_B1_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297467, upper bound: 5693.8297666
time: 0.84 seconds

## Relational analysis of NS_A2_A1_B1_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297467, upper bound: 5693.8297695
time: 0.69 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -3098.8637695, 2439.4809570, -3198.6035156, 2511.9204102, -5610.7841797, 5638.0844727
1: -260.2248535, 181.6754913, -268.2534790, 187.3858490, -447.6106262, 449.9289551
2: -176.3186646, 307.9901123, -181.4248199, 317.1923828, -493.5110474, 489.4148254
3: -214.6803284, 450.5090027, -221.0010986, 464.1204834, -678.8007812, 671.5100708
4: -172.5861511, 316.2916565, -177.4664001, 325.8106079, -498.3967590, 493.7580566

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8296587, upper bound: 5693.8297540
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A1_B1_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297656, upper bound: 5693.8297670
time: 0.57 seconds

## Relational analysis of NS_A2_A1_B1_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297656, upper bound: 5693.8297732
time: 0.56 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -3062.9389648, 2407.8356934, -3228.9916992, 2525.3710938, -5588.3100586, 5636.8261719
1: -256.9884338, 179.4144287, -269.8291321, 188.9623108, -445.9507446, 449.2435608
2: -173.8563690, 304.1032715, -183.2862549, 319.5431824, -493.3995056, 487.3894958
3: -211.7680054, 444.9633179, -223.1580505, 467.8480835, -679.6160278, 668.1213379
4: -170.2139130, 312.3470154, -179.3751831, 328.3226624, -498.5365601, 491.7221985

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A1_B1_B2_A1_A1

### Relational analysis result of NS_A2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297676, upper bound: 5693.8297623
time: 0.84 seconds

## Relational analysis of NS_A2_A1_B1_B2_A1_A2

### Relational analysis result of NS_A2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297676, upper bound: 5693.8297625
time: 0.58 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -3163.6564941, 2495.5725098, -3202.4763184, 2511.6542969, -5675.3105469, 5698.0483398
1: -266.1499329, 185.4952545, -268.1709290, 187.5052185, -453.6551514, 453.6661377
2: -179.9776611, 314.9075317, -181.9131317, 317.5216370, -497.4992981, 496.8206177
3: -219.2398224, 460.1109009, -221.5433960, 464.6961060, -683.9359131, 681.6542358
4: -176.0618286, 323.3519897, -178.0873871, 326.1550598, -502.2168884, 501.4393921

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A1_B1_B2_A2_A1

### Relational analysis result of NS_A2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297695, upper bound: 5693.8297670
time: 0.71 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2_A2

### Relational analysis result of NS_A2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297695, upper bound: 5693.8297728
time: 0.77 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -3048.1352539, 2397.9467773, -3151.5144043, 2471.6953125, -5519.8305664, 5549.4609375
1: -255.9248810, 178.5964966, -264.0641174, 184.6548309, -440.5797119, 442.6605530
2: -172.9996338, 302.7581787, -178.5797729, 312.2920532, -485.2916565, 481.3379517
3: -210.7343292, 443.1289673, -217.5129700, 457.2044678, -667.9387817, 660.6419678
4: -169.4416504, 310.9536438, -174.8431091, 320.7690430, -490.2106628, 485.7967529

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B2_B1_A1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8295897, upper bound: 5693.8297021
time: 0.67 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A1_B2_B1_A1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297478, upper bound: 5693.8297625
time: 0.58 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297478, upper bound: 5693.8297625
time: 0.67 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -3163.9831543, 2496.4841309, -3140.9785156, 2469.2941895, -5633.2773438, 5637.4628906
1: -266.2336121, 185.5940094, -263.6226501, 184.1242371, -450.3578491, 449.2166138
2: -180.0652466, 315.0144043, -178.2082977, 311.7762146, -491.8414612, 493.2226868
3: -219.3202209, 460.3161926, -217.0740662, 456.1857300, -675.5059814, 677.3902588
4: -176.1515808, 323.4580688, -174.4835663, 320.1840515, -496.3356323, 497.9416504

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B2_B1_A2_A1

### Relational analysis result of NS_A2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297348, upper bound: 5693.8297731
time: 0.79 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297720, upper bound: 5693.8297731
time: 0.80 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -3038.5715332, 2391.2390137, -3243.0332031, 2542.7949219, -5581.3657227, 5634.2724609
1: -255.1769104, 178.0127411, -271.6115112, 190.0475311, -445.2244263, 449.6242065
2: -172.4667969, 301.9188843, -183.9299469, 321.5181274, -493.9849243, 485.8488159
3: -210.1057892, 441.7555237, -224.0059814, 470.5284424, -680.6342163, 665.7613525
4: -168.9070129, 310.0751038, -180.0912628, 330.1940613, -499.1010742, 490.1663818

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A1_B2_B2_A1_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297676, upper bound: 5693.8297625
time: 0.57 seconds

## Relational analysis of NS_A2_A1_B2_B2_A1_A2

### Relational analysis result of NS_A2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297676, upper bound: 5693.8297625
time: 0.57 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -3152.1833496, 2488.6477051, -3231.1618652, 2539.8825684, -5692.0659180, 5719.8090820
1: -265.3579407, 184.8934631, -271.1180420, 189.4792175, -454.8371582, 456.0115051
2: -179.4198914, 314.0178223, -183.5251923, 320.9564209, -500.3762817, 497.5429688
3: -218.5646515, 458.6939392, -223.5323334, 469.3753967, -687.9400635, 682.2262573
4: -175.5265198, 322.4082336, -179.6963959, 329.5493164, -505.0758362, 502.1046143

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_A1_B2_B2_A2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297695, upper bound: 5693.8297671
time: 0.59 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297695, upper bound: 5693.8297729
time: 0.60 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -3583.2590332, 2820.4389648, -3113.8210449, 2447.2416992, -6030.5009766, 5934.2587891
1: -300.9514465, 210.9602203, -261.2484436, 182.4794769, -483.4309082, 472.2085876
2: -208.1096802, 357.7224121, -177.1178741, 309.0553589, -517.1650391, 534.8402710
3: -252.8248138, 521.2974243, -215.6376953, 452.2173462, -705.0420532, 736.9351196
4: -203.0989380, 366.8257141, -173.3584137, 317.4126892, -520.5115356, 540.1841431

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297526, upper bound: 5693.8297661
time: 0.58 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297624, upper bound: 5693.8297684
time: 0.79 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -3612.5764160, 2846.6052246, -3097.0908203, 2435.9265137, -6048.5029297, 5943.6943359
1: -303.6809998, 212.8774109, -260.0053711, 181.5324249, -485.2134094, 472.8827820
2: -209.6079712, 360.8516235, -176.1951599, 307.5916138, -517.1995850, 537.0466309
3: -254.6175842, 525.8991699, -214.5201263, 450.0350037, -704.6524048, 740.4193115
4: -204.5548096, 369.9717407, -172.4773254, 315.8789062, -520.4337158, 542.4490967

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297516, upper bound: 5693.8297627
time: 0.66 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297688, upper bound: 5693.8297727
time: 0.84 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -3566.0810547, 2810.5764160, -3195.6677246, 2511.9770508, -6078.0561523, 6006.2436523
1: -299.8060303, 209.9708252, -268.0632019, 187.2474213, -487.0534668, 478.0339355
2: -207.2503204, 356.4024658, -181.9498749, 317.4179077, -524.6681519, 538.3522949
3: -251.8263245, 519.1058960, -221.5247498, 464.1821899, -716.0084839, 740.6304932
4: -202.2777405, 365.4237061, -178.1003571, 325.9873962, -528.2651367, 543.5240479

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297696, upper bound: 5693.8297668
time: 0.70 seconds

## Relational analysis of NS_A2_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297729, upper bound: 5693.8297687
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -3595.4836426, 2836.8852539, -3171.5249023, 2495.8662109, -6091.3496094, 6008.4091797
1: -302.5492554, 211.8836365, -266.2928162, 185.8891296, -488.4383850, 478.1764221
2: -208.7525330, 359.5400085, -180.6323547, 315.3175659, -524.0700684, 540.1723633
3: -253.6229858, 523.7128296, -219.9427643, 461.0335999, -714.6565552, 743.6554565
4: -203.7333374, 368.5850220, -176.8568268, 323.7820740, -527.5153809, 545.4417725

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297688, upper bound: 5693.8297628
time: 0.57 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297727, upper bound: 5693.8297728
time: 0.72 seconds

## BFS NS instance: NS_A2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3581.0471191, 2821.2988281, -3584.8796387, 2824.0087891, -6405.0556641, 6406.1787109
1: -300.8927612, 210.8540802, -301.1920471, 211.0828857, -511.9756165, 512.0461426
2: -208.0160370, 357.9273376, -208.2203064, 358.2712402, -566.2871094, 566.1476440
3: -252.7596741, 521.2930298, -253.0086060, 521.8234863, -774.5830078, 774.3015747
4: -203.0079193, 366.9593506, -203.2056274, 367.3124084, -570.3203125, 570.1649170

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.7660403, upper bound: 5692.8643362
time: 0.78 seconds

## Relational analysis of NS_A2_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_A2_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -5692.8643362, upper bound: 5692.8643362
time: 0.71 seconds

## BFS NS instance: NS_A2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3581.0471191, 2821.2988281, -3812.0559082, 3003.9826660, -6585.0297852, 6633.3544922
1: -300.8927612, 210.8540802, -320.1646423, 225.2735291, -526.1662598, 531.0186157
2: -208.0160370, 357.9273376, -222.0696106, 381.2141724, -589.2301025, 579.9968872
3: -252.7596741, 521.2930298, -269.6117554, 555.3400879, -808.0996704, 790.9046631
4: -203.0079193, 366.9593506, -216.7897034, 390.9139404, -593.9217529, 583.7489624

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.7660403, upper bound: 5692.8643362
time: 0.65 seconds

## Relational analysis of NS_A2_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_A2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -5692.8643362, upper bound: 5692.8643362
time: 0.73 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3808.3254395, 3001.3354492, -3584.8796387, 2824.0087891, -6632.3339844, 6586.2148438
1: -319.8797607, 225.0525970, -301.1920471, 211.0828857, -530.9626465, 526.2445068
2: -221.8701630, 380.8742065, -208.2203064, 358.2712402, -580.1414185, 589.0944214
3: -269.3688965, 554.8281250, -253.0086060, 521.8234863, -791.1923828, 807.8366699
4: -216.5942535, 390.5656738, -203.2056274, 367.3124084, -583.9066772, 593.7711792

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297860, upper bound: 5693.8297841
time: 0.70 seconds

## Relational analysis of NS_A2_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297841, upper bound: 5693.8297838
time: 0.68 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3808.3254395, 3001.3354492, -3812.0559082, 3003.9826660, -6812.3081055, 6813.3916016
1: -319.8797607, 225.0525970, -320.1646423, 225.2735291, -545.1533203, 545.2170410
2: -221.8701630, 380.8742065, -222.0696106, 381.2141724, -603.0842896, 602.9437256
3: -269.3688965, 554.8281250, -269.6117554, 555.3400879, -824.7089233, 824.4397583
4: -216.5942535, 390.5656738, -216.7897034, 390.9139404, -607.5080566, 607.3552246

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8294913, upper bound: 5693.6827118
time: 0.72 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5692.8643362, upper bound: 5693.5640689
time: 0.83 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.39 seconds
NS_A1_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297352, upper bound: 5693.8297180
NS_A1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297352, upper bound: 5693.8297304
NS_A1_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297414, upper bound: 5693.8297180
NS_A1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297412, upper bound: 5693.8297363
NS_A1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297183, upper bound: 5693.8297183
NS_A1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297183, upper bound: 5693.8297183
NS_A1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297183, upper bound: 5693.8297183
NS_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297179, upper bound: 5693.8297309
NS_A1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297183, upper bound: 5693.8297183
NS_A1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297183, upper bound: 5693.8297183
NS_A1_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297183, upper bound: 5693.8297367
NS_A1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297183, upper bound: 5693.8297368
NS_A2_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297467, upper bound: 5693.8297666
NS_A2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297467, upper bound: 5693.8297695
NS_A2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297656, upper bound: 5693.8297670
NS_A2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297656, upper bound: 5693.8297732
NS_A2_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297676, upper bound: 5693.8297623
NS_A2_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297676, upper bound: 5693.8297625
NS_A2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297695, upper bound: 5693.8297670
NS_A2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297695, upper bound: 5693.8297728
NS_A2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297478, upper bound: 5693.8297625
NS_A2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297478, upper bound: 5693.8297625
NS_A2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297348, upper bound: 5693.8297731
NS_A2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297720, upper bound: 5693.8297731
NS_A2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297676, upper bound: 5693.8297625
NS_A2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297676, upper bound: 5693.8297625
NS_A2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297695, upper bound: 5693.8297671
NS_A2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297695, upper bound: 5693.8297729
NS_A2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297526, upper bound: 5693.8297661
NS_A2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297624, upper bound: 5693.8297684
NS_A2_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297516, upper bound: 5693.8297627
NS_A2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297688, upper bound: 5693.8297727
NS_A2_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297696, upper bound: 5693.8297668
NS_A2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297729, upper bound: 5693.8297687
NS_A2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297688, upper bound: 5693.8297628
NS_A2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297727, upper bound: 5693.8297728
NS_A2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.7660403, upper bound: 5692.8643362
NS_A2_A2_B2_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 5.39
Output dim: 0, lower bound: -5692.8643362, upper bound: 5692.8643362
NS_A2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.7660403, upper bound: 5692.8643362
NS_A2_A2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 5.39
Output dim: 0, lower bound: -5692.8643362, upper bound: 5692.8643362
NS_A2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297860, upper bound: 5693.8297841
NS_A2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8297841, upper bound: 5693.8297838
NS_A2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5693.8294913, upper bound: 5693.6827118
NS_A2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.39
Output dim: 0, lower bound: -5692.8643362, upper bound: 5693.5640689

## BFS NS instance: NS_A1_A2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -3829.4458008, 2939.8713379, -3617.2749023, 2778.7949219, -6608.2407227, 6557.1459961
1: -315.3671875, 226.0028381, -297.7424622, 211.0207062, -526.3878784, 523.7453003
2: -219.7993774, 375.7081299, -203.9456329, 353.1415100, -572.9408569, 579.6536865
3: -266.5244751, 552.5587769, -247.9740753, 518.9578247, -785.4822388, 800.5327759
4: -213.9724884, 386.1243591, -199.0031128, 363.3136902, -577.2861328, 585.1274414

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -3857.2810059, 2948.9089355, -3617.2749023, 2778.7949219, -6636.0751953, 6566.1835938
1: -316.7553711, 227.3869781, -297.7424622, 211.0207062, -527.7760620, 525.1293945
2: -221.2781830, 377.5567627, -203.9456329, 353.1415100, -574.4194946, 581.5023804
3: -268.4388428, 555.7319946, -247.9740753, 518.9578247, -787.3966675, 803.7060547
4: -215.5283203, 388.0203857, -199.0031128, 363.3136902, -578.8419800, 587.0234375

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -3829.4458008, 2939.8713379, -3648.2145996, 2785.5993652, -6615.0449219, 6588.0859375
1: -315.3671875, 226.0028381, -298.9281616, 212.3625183, -527.7297363, 524.9309692
2: -219.7993774, 375.7081299, -205.4341431, 354.4961548, -574.2955322, 581.1422729
3: -266.5244751, 552.5587769, -249.6739197, 521.7534180, -788.2778320, 802.2326660
4: -213.9724884, 386.1243591, -200.3029327, 364.7412109, -578.7135620, 586.4272461

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -3857.2810059, 2948.9089355, -3648.2145996, 2785.5993652, -6642.8803711, 6597.1235352
1: -316.7553711, 227.3869781, -298.9281616, 212.3625183, -529.1179199, 526.3150635
2: -221.2781830, 377.5567627, -205.4341431, 354.4961548, -575.7742310, 582.9909058
3: -268.4388428, 555.7319946, -249.6739197, 521.7534180, -790.1922607, 805.4058838
4: -215.5283203, 388.0203857, -200.3029327, 364.7412109, -580.2693481, 588.3232422

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_A2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3906.3347168, 2999.4653320, -3822.5866699, 2932.1940918, -6838.5268555, 6822.0517578
1: -321.7507019, 230.6793213, -314.5602722, 225.4411774, -547.1917725, 545.2395630
2: -224.6058655, 383.3260193, -219.2118225, 374.4782104, -599.0841064, 602.5377808
3: -272.2572937, 563.7211914, -265.7639465, 550.7535400, -823.0106812, 829.4851074
4: -218.5975037, 393.8852844, -213.1615906, 384.9403076, -603.5377808, 607.0468750

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

## BFS NS instance: NS_A1_A2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3906.3347168, 2999.4653320, -3890.6254883, 3025.6176758, -6931.9506836, 6890.0908203
1: -321.7507019, 230.6793213, -323.3306274, 229.2621460, -551.0128174, 554.0099487
2: -224.6058655, 383.3260193, -221.5864105, 383.6476440, -608.2534790, 604.9123535
3: -272.2572937, 563.7211914, -269.1620178, 563.6668091, -835.9240112, 832.8831787
4: -218.5975037, 393.8852844, -216.4513397, 394.6667175, -613.2641602, 610.3366089

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

## BFS NS instance: NS_A1_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3933.7758789, 3007.9921875, -3822.5866699, 2932.1940918, -6865.9682617, 6830.5781250
1: -323.0980530, 232.0151062, -314.5602722, 225.4411774, -548.5390625, 546.5753784
2: -226.0618286, 385.1126099, -219.2118225, 374.4782104, -600.5399170, 604.3244019
3: -274.1437988, 566.8295898, -265.7639465, 550.7535400, -824.8971558, 832.5935059
4: -220.1205597, 395.7281799, -213.1615906, 384.9403076, -605.0608521, 608.8896484

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

## BFS NS instance: NS_A1_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3933.7758789, 3007.9921875, -3890.6254883, 3025.6176758, -6959.3920898, 6898.6176758
1: -323.0980530, 232.0151062, -323.3306274, 229.2621460, -552.3601074, 555.3457031
2: -226.0618286, 385.1126099, -221.5864105, 383.6476440, -609.7094116, 606.6989746
3: -274.1437988, 566.8295898, -269.1620178, 563.6668091, -837.8104858, 835.9915771
4: -220.1205597, 395.7281799, -216.4513397, 394.6667175, -614.7872925, 612.1795044

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

## BFS NS instance: NS_A1_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3906.3347168, 2999.4653320, -3847.8510742, 2938.0871582, -6844.4194336, 6847.3154297
1: -321.7507019, 230.6793213, -315.6505127, 226.6615753, -548.4122314, 546.3295898
2: -224.6058655, 383.3260193, -220.4846191, 375.9474792, -600.5533447, 603.8105469
3: -272.2572937, 563.7211914, -267.4216614, 553.4357910, -825.6930542, 831.1427612
4: -218.5975037, 393.8852844, -214.5017395, 386.4441528, -605.0416260, 608.3870239

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

## BFS NS instance: NS_A1_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3906.3347168, 2999.4653320, -3919.3666992, 3033.1723633, -6939.5048828, 6918.8310547
1: -321.7507019, 230.6793213, -324.5915833, 230.6543121, -552.4050293, 555.2706909
2: -224.6058655, 383.3260193, -223.0659943, 385.2894592, -609.8953247, 606.3920288
3: -272.2572937, 563.7211914, -271.0520020, 566.6806641, -838.9379272, 834.7731934
4: -218.5975037, 393.8852844, -217.9632111, 396.3619690, -614.9594727, 611.8485107

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 45

## BFS NS instance: NS_A1_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3933.7758789, 3007.9921875, -3847.8510742, 2938.0871582, -6871.8613281, 6855.8427734
1: -323.0980530, 232.0151062, -315.6505127, 226.6615753, -549.7594604, 547.6654053
2: -226.0618286, 385.1126099, -220.4846191, 375.9474792, -602.0091553, 605.5971680
3: -274.1437988, 566.8295898, -267.4216614, 553.4357910, -827.5795288, 834.2512207
4: -220.1205597, 395.7281799, -214.5017395, 386.4441528, -606.5646973, 610.2297363

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

## BFS NS instance: NS_A1_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3933.7758789, 3007.9921875, -3919.3666992, 3033.1723633, -6966.9467773, 6927.3579102
1: -323.0980530, 232.0151062, -324.5915833, 230.6543121, -553.7523193, 556.6065063
2: -226.0618286, 385.1126099, -223.0659943, 385.2894592, -611.3510742, 608.1785889
3: -274.1437988, 566.8295898, -271.0520020, 566.6806641, -840.8244019, 837.8815918
4: -220.1205597, 395.7281799, -217.9632111, 396.3619690, -616.4825439, 613.6913452

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 45

## BFS NS instance: NS_A2_A1_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -3076.7846680, 2419.0686035, -3096.6042480, 2423.7744141, -5500.5585938, 5515.6718750
1: -258.2053833, 180.2662354, -259.0422058, 181.2160034, -439.4213257, 439.3084106
2: -174.8149567, 305.4462280, -175.3394623, 306.3741455, -481.1890259, 480.7857056
3: -212.8983917, 447.0456238, -213.5798798, 448.8952637, -661.7936401, 660.6254272
4: -171.1730652, 313.7138062, -171.6853943, 314.8195496, -485.9925842, 485.3992004

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B1_B1_B1_A1_A1

### Relational analysis result of NS_A2_A1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297448, upper bound: 5693.8297667
time: 0.58 seconds

## Relational analysis of NS_A2_A1_B1_B1_B1_A1_A2

### Relational analysis result of NS_A2_A1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297317, upper bound: 5693.8297634
time: 0.66 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -3087.0056152, 2434.2077637, -3096.6042480, 2423.7744141, -5510.7792969, 5530.8105469
1: -259.6833496, 181.0153961, -259.0422058, 181.2160034, -440.8993530, 440.0576172
2: -175.3341827, 307.1140747, -175.3394623, 306.3741455, -481.7082825, 482.4535217
3: -213.6040802, 449.1187744, -213.5798798, 448.8952637, -662.4992676, 662.6984863
4: -171.7548981, 315.3520813, -171.6853943, 314.8195496, -486.5744324, 487.0374146

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_B1_B1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297293, upper bound: 5693.8297635
time: 0.61 seconds

## Relational analysis of NS_A2_A1_B1_B1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297041, upper bound: 5693.8297635
time: 0.90 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -3061.9355469, 2412.5632324, -3198.6035156, 2511.9204102, -5573.8559570, 5611.1665039
1: -257.3358154, 179.4614258, -268.2534790, 187.3858490, -444.7215881, 447.7149048
2: -174.1327515, 304.4299927, -181.4248199, 317.1923828, -491.3251038, 485.8547974
3: -212.0853271, 445.3594055, -221.0010986, 464.1204834, -676.2057495, 666.3604736
4: -170.5393829, 312.6357727, -177.4664001, 325.8106079, -496.3499756, 490.1021729

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B1_B1_B2_A1_A1

### Relational analysis result of NS_A2_A1_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297639, upper bound: 5693.8297668
time: 0.67 seconds

## Relational analysis of NS_A2_A1_B1_B1_B2_A1_A2

### Relational analysis result of NS_A2_A1_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297460, upper bound: 5693.8297633
time: 0.59 seconds

## BFS NS instance: NS_A2_A1_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -3088.8327637, 2440.0793457, -3198.6035156, 2511.9204102, -5600.7529297, 5638.6816406
1: -260.1385193, 181.2276001, -268.2534790, 187.3858490, -447.5243225, 449.4810791
2: -175.7262268, 307.7713623, -181.4248199, 317.1923828, -492.9185181, 489.1961365
3: -214.0740356, 449.7926941, -221.0010986, 464.1204834, -678.1944580, 670.7938232
4: -172.1346283, 315.9598694, -177.4664001, 325.8106079, -497.9451599, 493.4262695

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B1_B1_B2_A2_A1

### Relational analysis result of NS_A2_A1_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297639, upper bound: 5693.8297711
time: 0.67 seconds

## Relational analysis of NS_A2_A1_B1_B1_B2_A2_A2

### Relational analysis result of NS_A2_A1_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297460, upper bound: 5693.8297633
time: 0.59 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -3031.2797852, 2384.7753906, -3228.9916992, 2525.3710938, -5556.6508789, 5613.7666016
1: -254.5061188, 177.5023193, -269.8291321, 188.9623108, -443.4684448, 447.3314514
2: -171.9907074, 301.0435791, -183.2862549, 319.5431824, -491.5338745, 484.3298035
3: -209.5477753, 440.5202637, -223.1580505, 467.8480835, -677.3958740, 663.6782837
4: -168.4606781, 309.1971741, -179.3751831, 328.3226624, -496.7833252, 488.5723572

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B1_B2_A1_A1_A1

### Relational analysis result of NS_A2_A1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297668, upper bound: 5693.8297612
time: 0.58 seconds

## Relational analysis of NS_A2_A1_B1_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_A1_B1_B2_A1_A1_A1

### Relational analysis result of NS_A2_A1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297455, upper bound: 5693.8297216
time: 0.70 seconds

## Relational analysis of NS_A2_A1_B1_B2_A1_A1_A2

### Relational analysis result of NS_A2_A1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297455, upper bound: 5693.8297625
time: 0.70 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -3034.8608398, 2394.7280273, -3228.9916992, 2525.3710938, -5560.2319336, 5623.7197266
1: -255.4505310, 177.8452759, -269.8291321, 188.9623108, -444.4128418, 447.6744080
2: -172.0738678, 302.0415649, -183.2862549, 319.5431824, -491.6170349, 485.3277588
3: -209.7431335, 441.6944275, -223.1580505, 467.8480835, -677.5911865, 664.8524780
4: -168.6518707, 310.1369324, -179.3751831, 328.3226624, -496.9745178, 489.5121155

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B1_B2_A1_A2_A1

### Relational analysis result of NS_A2_A1_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297668, upper bound: 5693.8297612
time: 0.64 seconds

## Relational analysis of NS_A2_A1_B1_B2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_A1_B1_B2_A1_A2_A1

### Relational analysis result of NS_A2_A1_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297455, upper bound: 5693.8297217
time: 0.70 seconds

## Relational analysis of NS_A2_A1_B1_B2_A1_A2_A2

### Relational analysis result of NS_A2_A1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297455, upper bound: 5693.8297626
time: 0.89 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -3127.8725586, 2469.0629883, -3202.4763184, 2511.6542969, -5639.5253906, 5671.5390625
1: -263.3194885, 183.3245697, -268.1709290, 187.5052185, -450.8247070, 451.4954834
2: -177.8327332, 311.3995972, -181.9131317, 317.5216370, -495.3543091, 493.3126831
3: -216.6913300, 455.0650330, -221.5433960, 464.6961060, -681.3874512, 676.6083984
4: -174.0504761, 319.7520142, -178.0873871, 326.1550598, -500.2055359, 497.8394165

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B1_B2_A2_A1_A1

### Relational analysis result of NS_A2_A1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297679, upper bound: 5693.8297669
time: 0.59 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2_A1_A2

### Relational analysis result of NS_A2_A1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297671, upper bound: 5693.8297633
time: 0.66 seconds

## BFS NS instance: NS_A2_A1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -3189.5908203, 2518.9160156, -3202.4763184, 2511.6542969, -5701.2451172, 5721.3925781
1: -268.6415710, 187.1936646, -268.1709290, 187.5052185, -456.1467896, 455.3645630
2: -181.2633972, 317.6674500, -181.9131317, 317.5216370, -498.7850342, 499.5805664
3: -220.8354340, 464.1873474, -221.5433960, 464.6961060, -685.5314941, 685.7307129
4: -177.3983917, 326.0586853, -178.0873871, 326.1550598, -503.5534668, 504.1460571

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_A1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A2_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297647, upper bound: 5693.8297713
time: 0.57 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A2_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297672, upper bound: 5693.8297630
time: 0.67 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2_A2_B2

### Relational analysis result of NS_A2_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297665, upper bound: 5693.8297633
time: 0.86 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -3038.7414551, 2390.0893555, -3151.5144043, 2471.6953125, -5510.4365234, 5541.6035156
1: -255.1005402, 177.9648743, -264.0641174, 184.6548309, -439.7553711, 442.0289917
2: -172.4073181, 301.6993713, -178.5797729, 312.2920532, -484.6993713, 480.2791443
3: -210.0409088, 441.6172485, -217.5129700, 457.2044678, -667.2453613, 659.1302490
4: -168.8863220, 309.8839417, -174.8431091, 320.7690430, -489.6553650, 484.7270508

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_A1_B2_B1_A1_A1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297453, upper bound: 5693.8297216
time: 0.61 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_A1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -5693.8297453, upper bound: 5693.8297625
time: 0.60 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -3046.7287598, 2403.2221680, -3151.5144043, 2471.6953125, -5518.4238281, 5554.7363281
1: -256.3825378, 178.5701447, -264.0641174, 184.6548309, -441.0373535, 442.6342773
2: -172.7562714, 303.1168823, -178.5797729, 312.2920532, -485.0483093, 481.6966553
3: -210.5451355, 443.3832397, -217.5129700, 457.2044678, -667.7495728, 660.8962402
4: -169.3164062, 311.2577209, -174.8431091, 320.7690430, -490.0854492, 486.1008301

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.90 + 415.35 = 420.25 seconds
