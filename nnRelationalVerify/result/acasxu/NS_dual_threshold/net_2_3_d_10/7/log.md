## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 465.361891711094


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-141.4966431, 408.7326050, -141.4966431, 408.7326050, -550.2292480, 550.2292480)
1: (-100.5886383, 256.2147217, -100.5886383, 256.2147217, -356.8033447, 356.8033447)
2: (-110.1150436, 236.7749176, -110.1150436, 236.7749176, -346.8898621, 346.8898621)
3: (-99.3735046, 306.2141724, -99.3735046, 306.2141724, -405.5876465, 405.5876465)
4: (-158.7235718, 250.7310791, -158.7235718, 250.7310791, -409.4546509, 409.4546509)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.68 + 2.13 = 3.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -465.3898151, upper bound: 465.3898151

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3642102, upper bound: 465.3799891
time: 0.88 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3636018, upper bound: 465.3636018
time: 0.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.96 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.96
Output dim: 0, lower bound: -465.3642102, upper bound: 465.3799891
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.96
Output dim: 0, lower bound: -465.3636018, upper bound: 465.3636018

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -140.5482483, 405.8926392, -141.4966431, 408.7326050, -549.2808838, 547.3892212
1: -99.9007721, 254.4438782, -100.5886383, 256.2147217, -356.1154175, 355.0325012
2: -109.3579559, 235.1018372, -110.1150436, 236.7749176, -346.1327820, 345.2168274
3: -98.6999588, 304.1059265, -99.3735046, 306.2141724, -404.9140625, 403.4794312
4: -157.6277771, 248.9960327, -158.7235718, 250.7310791, -408.3588562, 407.7196045

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3636018, upper bound: 465.3636018
time: 0.80 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3636018, upper bound: 465.3636018
time: 1.10 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -161.1155243, 459.7918701, -140.5012207, 405.8188477, -566.9343262, 600.2929688
1: -113.9622269, 287.9673157, -99.8776627, 254.3805695, -368.3427734, 387.8449707
2: -124.7284317, 264.9818726, -109.3470993, 235.0447845, -359.7732239, 374.3289490
3: -112.5406113, 345.3304138, -98.6817703, 304.0445557, -416.5851440, 444.0121460
4: -178.6892090, 282.4119263, -157.6147003, 248.9411011, -427.6301880, 440.0266113

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3629487, upper bound: 465.3602469
time: 0.64 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451
time: 0.84 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.38 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.38
Output dim: 0, lower bound: -465.3636018, upper bound: 465.3636018
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.38
Output dim: 0, lower bound: -465.3636018, upper bound: 465.3636018
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.38
Output dim: 0, lower bound: -465.3629487, upper bound: 465.3602469
NS_A2_B2, status: Status.VERIFIED, split count: 2, time: 5.38
Output dim: 0, lower bound: -465.3598451, upper bound: 465.3598451

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -140.5482483, 405.8926392, -140.5482483, 405.8926392, -546.4409180, 546.4409180
1: -99.9007721, 254.4438782, -99.9007721, 254.4438782, -354.3446045, 354.3445740
2: -109.3579559, 235.1018372, -109.3579559, 235.1018372, -344.4597473, 344.4597473
3: -98.6999588, 304.1059265, -98.6999588, 304.1059265, -402.8058472, 402.8058472
4: -157.6277771, 248.9960327, -157.6277771, 248.9960327, -406.6238098, 406.6238098

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3615587, upper bound: 465.3660981
time: 1.04 seconds

## Relational analysis of NS_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3631638, upper bound: 465.3712032
time: 0.82 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3601878, upper bound: 465.3712050
time: 0.60 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -140.5482483, 405.8926392, -161.1155243, 459.7918701, -600.3400879, 567.0079956
1: -99.9007721, 254.4438782, -113.9622269, 287.9673157, -387.8680115, 368.4060669
2: -109.3579559, 235.1018372, -124.7284317, 264.9818726, -374.3398132, 359.8302307
3: -98.6999588, 304.1059265, -112.5406113, 345.3304138, -444.0303040, 416.6464844
4: -157.6277771, 248.9960327, -178.6892090, 282.4119263, -440.0397034, 427.6852112

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3615587, upper bound: 465.3660981
time: 0.65 seconds

## Relational analysis of NS_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3608832, upper bound: 465.3796354
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3601878, upper bound: 465.3712050
time: 0.94 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -161.1155243, 459.7918701, -133.6057281, 386.5313721, -547.6465454, 593.3975830
1: -113.9622269, 287.9673157, -95.0613785, 242.1999207, -356.1621399, 383.0286255
2: -124.7284317, 264.9818726, -104.2362900, 223.8220062, -348.5503845, 369.2181396
3: -112.5406113, 345.3304138, -93.9838257, 289.6250610, -402.1656494, 439.3142395
4: -178.6892090, 282.4119263, -150.2775116, 237.0852051, -415.7743835, 432.6894226

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3624578, upper bound: 465.3598835
time: 0.64 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3626937, upper bound: 465.3599056
time: 0.64 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.58 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.58
Output dim: 0, lower bound: -465.3631638, upper bound: 465.3712032
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 5.58
Output dim: 0, lower bound: -465.3601878, upper bound: 465.3712050
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 5.58
Output dim: 0, lower bound: -465.3608832, upper bound: 465.3796354
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 5.58
Output dim: 0, lower bound: -465.3601878, upper bound: 465.3712050
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 5.58
Output dim: 0, lower bound: -465.3624578, upper bound: 465.3598835
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 5.58
Output dim: 0, lower bound: -465.3626937, upper bound: 465.3599056

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -140.5482483, 405.8926392, -133.7652740, 386.9566345, -527.5048828, 539.6578979
1: -99.9007721, 254.4438782, -95.1698761, 242.4745941, -342.3752747, 349.6137390
2: -109.3579559, 235.1018372, -104.3475342, 224.0744171, -333.4322815, 339.4493713
3: -98.6999588, 304.1059265, -94.0874939, 289.9502869, -388.6501770, 398.1934204
4: -157.6277771, 248.9960327, -150.4336853, 237.3517151, -394.9794922, 399.4297180

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3733913, upper bound: 465.3733913
time: 0.72 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3733913, upper bound: 465.3733913
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -136.5247498, 394.2730103, -177.5588989, 513.1923218, -649.6176758, 571.8318481
1: -97.0526657, 247.2702789, -126.8906479, 321.9714661, -419.0241089, 374.1084595
2: -106.3021240, 228.4709778, -138.9537354, 297.8376465, -404.1397705, 367.4247131
3: -95.9171448, 295.5176392, -125.1197281, 385.3390503, -481.2561646, 420.6373596
4: -153.2145844, 241.9704132, -199.3551178, 315.8643494, -469.0789185, 441.3254395

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3733913, upper bound: 465.3733913
time: 0.77 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3733913, upper bound: 465.3733913
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -133.7652740, 386.9566345, -161.1155243, 459.7918701, -593.5570679, 548.0720825
1: -95.1698761, 242.4745941, -113.9622269, 287.9673157, -383.1372070, 356.4367676
2: -104.3475342, 224.0744171, -124.7284317, 264.9818726, -369.3294067, 348.8027954
3: -94.0874939, 289.9502869, -112.5406113, 345.3304138, -439.4179077, 402.4908752
4: -150.4336853, 237.3517151, -178.6892090, 282.4119263, -432.8456116, 416.0408630

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3605743, upper bound: 465.3794558
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3605712, upper bound: 465.3794998
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -177.5588989, 513.1923218, -157.1955872, 448.6198425, -626.1787109, 669.9376831
1: -126.8906479, 321.9714661, -111.2154617, 281.0154724, -407.6879578, 433.1868591
2: -138.9537354, 297.8376465, -121.8285522, 258.5561218, -397.4943848, 419.6661682
3: -125.1197281, 385.3390503, -109.8706818, 337.0618896, -462.1816101, 495.1795959
4: -199.3551178, 315.8643494, -174.5041656, 275.6260376, -474.8229980, 490.3685303

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3601878, upper bound: 465.3712032
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3601878, upper bound: 465.3712050
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -158.7193756, 452.9590454, -126.8152924, 367.3130798, -526.0323486, 579.7742310
1: -112.2942581, 283.6403809, -90.3530655, 229.9905090, -342.2846985, 373.9933777
2: -122.9298782, 260.9397278, -99.1435928, 212.3939056, -335.3237915, 360.0832825
3: -110.8988113, 340.2354736, -89.3609009, 275.2838135, -386.1825867, 429.5963745
4: -176.0864716, 278.2116394, -142.9349823, 225.2440186, -401.3304749, 421.1466064

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3624578, upper bound: 465.3598835
time: 0.80 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3624578, upper bound: 465.3598835
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -159.0547485, 453.5940552, -132.1165771, 381.9939575, -541.0486450, 585.7106323
1: -112.5007172, 284.1868591, -94.0480957, 239.3257904, -351.8265076, 378.2349548
2: -123.1155701, 261.4404602, -103.1971436, 221.1005096, -344.2160645, 364.6375732
3: -111.1029587, 340.7877197, -93.0013046, 286.3482056, -397.4511719, 433.7890320
4: -176.3394165, 278.6869202, -148.6488037, 234.3498383, -410.6892395, 427.3356323

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3626937, upper bound: 465.3599056
time: 0.70 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3626937, upper bound: 465.3599056
time: 0.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.08 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -465.3733913, upper bound: 465.3733913
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -465.3733913, upper bound: 465.3733913
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -465.3733913, upper bound: 465.3733913
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -465.3733913, upper bound: 465.3733913
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -465.3605743, upper bound: 465.3794558
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -465.3605712, upper bound: 465.3794998
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -465.3601878, upper bound: 465.3712032
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -465.3601878, upper bound: 465.3712050
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -465.3624578, upper bound: 465.3598835
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -465.3624578, upper bound: 465.3598835
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -465.3626937, upper bound: 465.3599056
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.08
Output dim: 0, lower bound: -465.3626937, upper bound: 465.3599056

## BFS NS instance: NS_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -133.7652740, 386.9566345, -133.7652740, 386.9566345, -520.7219238, 520.7219238
1: -95.1698761, 242.4745941, -95.1698761, 242.4745941, -337.6444702, 337.6444702
2: -104.3475342, 224.0744171, -104.3475342, 224.0744171, -328.4219055, 328.4219055
3: -94.0874939, 289.9502869, -94.0874939, 289.9502869, -384.0377808, 384.0377808
4: -150.4336853, 237.3517151, -150.4336853, 237.3517151, -387.7854004, 387.7854004

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3852051, upper bound: 465.3740992
time: 0.75 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3855911, upper bound: 465.3741188
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -177.5588989, 513.1923218, -133.7652740, 386.9566345, -564.5155029, 646.8621826
1: -126.8906479, 321.9714661, -95.1698761, 242.4745941, -369.2970276, 417.1413269
2: -138.9537354, 297.8376465, -104.3475342, 224.0744171, -363.0281372, 402.1851807
3: -125.1197281, 385.3390503, -94.0874939, 289.9502869, -415.0700073, 479.4265442
4: -199.3551178, 315.8643494, -150.4336853, 237.3517151, -436.7068176, 466.2980347

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3856211, upper bound: 465.3741168
time: 0.83 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3845142, upper bound: 465.3740852
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -133.7652740, 386.9566345, -177.5588989, 513.1923218, -646.8621826, 564.5155029
1: -95.1698761, 242.4745941, -126.8906479, 321.9714661, -417.1413269, 369.2970276
2: -104.3475342, 224.0744171, -138.9537354, 297.8376465, -402.1851807, 363.0281372
3: -94.0874939, 289.9502869, -125.1197281, 385.3390503, -479.4265442, 415.0700073
4: -150.4336853, 237.3517151, -199.3551178, 315.8643494, -466.2980347, 436.7068176

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3695995, upper bound: 465.3718021
time: 0.66 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3729957, upper bound: 465.3729957
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -177.5588989, 513.1923218, -177.5588989, 513.1923218, -690.1534424, 690.1534424
1: -126.8906479, 321.9714661, -126.8906479, 321.9714661, -448.7399902, 448.7399902
2: -138.9537354, 297.8376465, -138.9537354, 297.8376465, -436.7913818, 436.7913818
3: -125.1197281, 385.3390503, -125.1197281, 385.3390503, -510.2675171, 510.2675171
4: -199.3551178, 315.8643494, -199.3551178, 315.8643494, -515.1195679, 515.1195679

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B2_A2_B1

### Relational analysis result of NS_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3708362, upper bound: 465.3724417
time: 0.70 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2

### Relational analysis result of NS_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3727709, upper bound: 465.3727709
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -126.9808121, 367.7620544, -158.7193756, 452.9590454, -579.9398193, 526.4814453
1: -90.4662857, 230.2761536, -112.2942581, 283.6403809, -374.1066589, 342.5702820
2: -99.2604752, 212.6578979, -122.9298782, 260.9397278, -360.2001648, 335.5877686
3: -89.4693222, 275.6247559, -110.8988113, 340.2354736, -429.7047729, 386.5235596
4: -143.0995178, 225.5220642, -176.0864716, 278.2116394, -421.3111267, 401.6085205

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3605743, upper bound: 465.3794555
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3605743, upper bound: 465.3794558
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -132.4024658, 382.8420715, -159.0547485, 453.5940552, -585.9963989, 541.8968506
1: -94.2508698, 239.8445587, -112.5007172, 284.1868591, -378.4377441, 352.3452759
2: -103.4209290, 221.5840149, -123.1155701, 261.4404602, -364.8613892, 344.6995850
3: -93.2020340, 286.9747314, -111.1029587, 340.7877197, -433.9897461, 398.0776978
4: -148.9710999, 234.8580780, -176.3394165, 278.6869202, -427.6579895, 411.1974792

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3605712, upper bound: 465.3794907
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3605712, upper bound: 465.3794998
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -177.5588989, 513.1923218, -153.8293610, 439.1490173, -616.7078857, 666.5657959
1: -126.8906479, 321.9714661, -108.8408890, 274.9800720, -401.6484070, 430.8123474
2: -138.9537354, 297.8376465, -119.2062149, 253.0222015, -391.9565735, 417.0438538
3: -125.1197281, 385.3390503, -107.5071564, 329.9026184, -455.0222778, 492.8134155
4: -199.3551178, 315.8643494, -170.7487640, 269.7779236, -468.9692383, 486.6130676

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3581556, upper bound: 465.3555586
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3595697, upper bound: 465.3703455
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -177.5588989, 513.1923218, -199.0057373, 569.0985718, -745.4471436, 711.1714478
1: -126.8906479, 321.9714661, -141.5508728, 356.6902466, -483.1954651, 463.2090759
2: -138.9537354, 297.8376465, -154.9951935, 328.4560547, -467.3139954, 452.6310730
3: -125.1197281, 385.3390503, -139.6377258, 428.1783447, -552.8039551, 524.6953125
4: -199.3551178, 315.8643494, -221.4843903, 350.3638000, -549.4358521, 537.0452881

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -465.3581556, upper bound: 465.3566569
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3595697, upper bound: 465.3703455
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -151.5298309, 432.6177063, -126.8152924, 367.3130798, -518.8428345, 559.4329834
1: -107.2448425, 270.8350220, -90.3530655, 229.9905090, -337.2353516, 361.1880493
2: -117.4931641, 249.1458130, -99.1435928, 212.3939056, -329.8870850, 348.2893982
3: -105.9395828, 325.0317383, -89.3609009, 275.2838135, -381.2233276, 414.3926392
4: -168.2697754, 265.7545471, -142.9349823, 225.2440186, -393.5137024, 408.6895142

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3624578, upper bound: 465.3598835
time: 0.78 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3624578, upper bound: 465.3598835
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -197.3084412, 564.2221680, -126.8152924, 367.3130798, -564.6215210, 690.2833252
1: -140.3680878, 353.6281128, -90.3530655, 229.9905090, -370.0820007, 443.9811707
2: -153.7138672, 325.5948486, -99.1435928, 212.3939056, -365.9310913, 424.7384338
3: -138.4787903, 424.5709534, -89.3609009, 275.2838135, -413.7625427, 513.6427612
4: -219.6260223, 347.3887634, -142.9349823, 225.2440186, -444.6454773, 490.3237305

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3624578, upper bound: 465.3598835
time: 0.80 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3624578, upper bound: 465.3598835
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -151.7067871, 432.6956787, -132.1165771, 381.9939575, -533.7006226, 564.8122559
1: -107.3257980, 271.0708008, -94.0480957, 239.3257904, -346.6514893, 365.1188965
2: -117.5263596, 249.3552856, -103.1971436, 221.1005096, -338.6268311, 352.5523987
3: -106.0172272, 325.1901550, -93.0013046, 286.3482056, -392.3653870, 418.1914673
4: -168.3002014, 265.9169922, -148.6488037, 234.3498383, -402.6500244, 414.5656738

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3626937, upper bound: 465.3599056
time: 0.79 seconds

## Relational analysis of NS_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3626937, upper bound: 465.3599056
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -195.5906372, 559.0335083, -132.1165771, 381.9939575, -577.5844116, 690.4939575
1: -139.1471405, 350.4203796, -94.0480957, 239.3257904, -378.2258606, 444.4684753
2: -152.3687286, 322.5435791, -103.1971436, 221.1005096, -373.3124695, 425.7406921
3: -137.2671204, 420.7194519, -93.0013046, 286.3482056, -423.6152649, 513.4605103
4: -217.6697235, 344.2184143, -148.6488037, 234.3498383, -451.8361206, 492.8671265

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3626937, upper bound: 465.3599056
time: 0.76 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3626937, upper bound: 465.3599056
time: 0.71 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.00 seconds
NS_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3852051, upper bound: 465.3740992
NS_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3855911, upper bound: 465.3741188
NS_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3856211, upper bound: 465.3741168
NS_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3845142, upper bound: 465.3740852
NS_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3695995, upper bound: 465.3718021
NS_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3729957, upper bound: 465.3729957
NS_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3708362, upper bound: 465.3724417
NS_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3727709, upper bound: 465.3727709
NS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3605743, upper bound: 465.3794555
NS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3605743, upper bound: 465.3794558
NS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3605712, upper bound: 465.3794907
NS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3605712, upper bound: 465.3794998
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3581556, upper bound: 465.3555586
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3595697, upper bound: 465.3703455
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3581556, upper bound: 465.3566569
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3595697, upper bound: 465.3703455
NS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3624578, upper bound: 465.3598835
NS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3624578, upper bound: 465.3598835
NS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3624578, upper bound: 465.3598835
NS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3624578, upper bound: 465.3598835
NS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3626937, upper bound: 465.3599056
NS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3626937, upper bound: 465.3599056
NS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3626937, upper bound: 465.3599056
NS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.00
Output dim: 0, lower bound: -465.3626937, upper bound: 465.3599056

## BFS NS instance: NS_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -131.1943665, 379.8722839, -125.0170975, 362.7051392, -493.8995056, 504.8893738
1: -93.3984146, 237.9232941, -89.1052780, 226.9123383, -320.3107300, 327.0285034
2: -102.4822845, 219.8931885, -97.9602280, 209.7592468, -312.2415161, 317.8534241
3: -92.3624725, 284.6224670, -88.1813278, 271.7160950, -364.0785522, 372.8038025
4: -147.7152100, 232.9754639, -141.1258545, 222.3691559, -370.0843506, 374.1013184

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3874882, upper bound: 465.3874882
time: 0.85 seconds

## Relational analysis of NS_A1_B1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3874882, upper bound: 465.3874882
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -132.5899658, 383.2510071, -152.6236572, 446.5347595, -578.9328003, 535.8746338
1: -94.3165588, 240.2841339, -109.3754349, 278.4595642, -372.7760620, 349.6595459
2: -103.4097900, 222.0371094, -119.6653824, 257.4065552, -360.8163452, 341.7024841
3: -93.2556686, 287.2797852, -107.9981232, 333.4006958, -426.6563721, 395.2778931
4: -149.0198212, 235.2120056, -172.8643799, 273.0295715, -422.0493774, 408.0763245

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3844308, upper bound: 465.3871392
time: 0.80 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3844308, upper bound: 465.3874848
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -176.7545471, 510.9248657, -124.4543076, 361.1313477, -537.8858032, 635.2636108
1: -126.3334198, 320.5410156, -88.7355270, 225.9476013, -352.1948853, 409.2765198
2: -138.3593445, 296.5104980, -97.5271225, 208.7811737, -347.1405029, 394.0376282
3: -124.5783234, 383.6551514, -87.8218918, 270.5800781, -395.1583862, 471.4770508
4: -198.4914093, 314.4806519, -140.5474396, 221.3910522, -419.8721008, 455.0280762

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3848652, upper bound: 465.3741114
time: 0.67 seconds

## Relational analysis of NS_A1_B1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3844025, upper bound: 465.3740866
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -173.2501678, 500.5060730, -165.8697662, 483.6058655, -655.0912476, 665.8649902
1: -123.7850571, 314.0972290, -118.7599716, 301.9506531, -425.4223938, 432.3959961
2: -135.5907745, 290.4692688, -129.9200592, 278.1240845, -413.7148438, 420.3851013
3: -122.0786667, 375.9143982, -117.2787399, 361.5428162, -483.1480713, 493.0264587
4: -194.4619904, 308.1046753, -187.2673187, 295.9468079, -490.2636108, 494.9027405

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3773314, upper bound: 465.3737561
time: 0.67 seconds

## Relational analysis of NS_A1_B1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3842744, upper bound: 465.3740648
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -131.5792389, 380.5763550, -171.0844574, 494.3450012, -625.7274170, 551.6608276
1: -93.6153946, 238.5114288, -122.2808456, 310.2258301, -403.8411865, 360.6672058
2: -102.6221313, 220.3843079, -133.7320557, 286.9319763, -389.5541077, 354.1163635
3: -92.5473557, 285.1986084, -120.5106659, 371.2304382, -463.7656555, 405.7092896
4: -147.9704437, 233.4529877, -191.9670715, 304.2737732, -452.2442017, 425.3993835

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3687168, upper bound: 465.3775954
time: 0.70 seconds

## Relational analysis of NS_A1_B1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3695046, upper bound: 465.3846475
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -133.7652740, 386.9566345, -176.8706665, 511.2015686, -644.8856812, 563.8272705
1: -95.1698761, 242.4745941, -126.4033890, 320.7334595, -415.9033203, 368.8145752
2: -104.3475342, 224.0744171, -138.4208679, 296.6872253, -401.0347595, 362.4952393
3: -94.0874939, 289.9502869, -124.6389389, 383.8566589, -477.9441528, 414.5892334
4: -150.4336853, 237.3517151, -198.5930328, 314.6499329, -465.0836182, 435.9447327

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3736127, upper bound: 465.3814737
time: 0.82 seconds

## Relational analysis of NS_A1_B1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3737278, upper bound: 465.3853606
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -174.4510040, 503.2310181, -171.9126892, 493.9064636, -667.6880493, 674.1945801
1: -124.5560532, 315.9988708, -122.4988861, 310.6325684, -434.9716797, 438.3670349
2: -136.3942719, 292.2113037, -134.0368500, 287.0077515, -423.3570862, 426.2350769
3: -122.8388367, 378.1283569, -120.8424225, 371.6581116, -494.2821045, 498.7022095
4: -195.6140747, 309.9208374, -192.1724548, 304.4881592, -499.9096680, 501.9843750

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3705253, upper bound: 465.3705253
time: 0.81 seconds

## Relational analysis of NS_A1_B1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3705253, upper bound: 465.3724417
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -177.5588989, 513.1923218, -176.9825134, 511.4635620, -688.4304199, 689.5764160
1: -126.8906479, 321.9714661, -126.4742737, 320.9034424, -447.6699524, 448.3266296
2: -138.9537354, 297.8376465, -138.5057678, 296.8290100, -435.7827454, 436.3434143
3: -125.1197281, 385.3390503, -124.7135239, 384.0748901, -509.0043030, 509.8613281
4: -199.3551178, 315.8643494, -198.7013702, 314.8189697, -514.0726318, 514.4678345

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3724417, upper bound: 465.3708362
time: 1.08 seconds

## Relational analysis of NS_A1_B1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3724417, upper bound: 465.3727709
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -126.9808121, 367.7620544, -151.5298309, 432.6177063, -559.5985107, 519.2918701
1: -90.4662857, 230.2761536, -107.2448425, 270.8350220, -361.3013000, 337.5209961
2: -99.2604752, 212.6578979, -117.4931641, 249.1458130, -348.4062805, 330.1510620
3: -89.4693222, 275.6247559, -105.9395828, 325.0317383, -414.5010071, 381.5643311
4: -143.0995178, 225.5220642, -168.2697754, 265.7545471, -408.8540039, 393.7917786

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

## BFS NS instance: NS_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -126.9808121, 367.7620544, -197.3084412, 564.2221680, -690.4454956, 565.0704956
1: -90.4662857, 230.2761536, -140.3680878, 353.6281128, -444.0943909, 370.3646240
2: -99.2604752, 212.6578979, -153.7138672, 325.5948486, -424.8553162, 366.1944885
3: -89.4693222, 275.6247559, -138.4787903, 424.5709534, -513.7514038, 414.1035461
4: -143.0995178, 225.5220642, -219.6260223, 347.3887634, -490.4882507, 444.9217834

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 23

## BFS NS instance: NS_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -132.4024658, 382.8420715, -151.7067871, 432.6956787, -565.0980835, 534.5488281
1: -94.2508698, 239.8445587, -107.3257980, 271.0708008, -365.3216553, 347.1702271
2: -103.4209290, 221.5840149, -117.5263596, 249.3552856, -352.7762146, 339.1103210
3: -93.2020340, 286.9747314, -106.0172272, 325.1901550, -418.3921814, 392.9919434
4: -148.9710999, 234.8580780, -168.3002014, 265.9169922, -414.8880310, 403.1582642

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3586119, upper bound: 465.3707358
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3586119, upper bound: 465.3789376
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -132.4024658, 382.8420715, -195.5906372, 559.0335083, -690.7758789, 578.4326782
1: -94.2508698, 239.8445587, -139.1471405, 350.4203796, -444.6712341, 378.7415771
2: -103.4209290, 221.5840149, -152.3687286, 322.5435791, -425.9645081, 373.7952576
3: -93.2020340, 286.9747314, -137.2671204, 420.7194519, -513.6606445, 424.2418213
4: -148.9710999, 234.8580780, -217.6697235, 344.2184143, -493.1894836, 452.3428650

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 23

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -176.9825134, 511.4635620, -153.8293610, 439.1490173, -616.1315308, 664.8427734
1: -126.4742737, 320.9034424, -108.8408890, 274.9800720, -401.2350159, 429.7443237
2: -138.5057678, 296.8290100, -119.2062149, 253.0222015, -391.5095215, 416.0352173
3: -124.7135239, 384.0748901, -107.5071564, 329.9026184, -454.6160889, 491.5502319
4: -198.7013702, 314.8189697, -170.7487640, 269.7779236, -468.3174438, 485.5676880

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3605654, upper bound: 465.3699291
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -465.3608368, upper bound: 465.3703387
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -176.9825134, 511.4635620, -199.0057373, 569.0985718, -744.8700562, 709.4484253
1: -126.4742737, 320.9034424, -141.5508728, 356.6902466, -482.7820740, 462.1390686
2: -138.5057678, 296.8290100, -154.9951935, 328.4560547, -466.8669128, 451.6204834
3: -124.7135239, 384.0748901, -139.6377258, 428.1783447, -552.3977661, 523.4321899
4: -198.7013702, 314.8189697, -221.4843903, 350.3638000, -548.7840576, 535.9983521

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

## BFS NS instance: NS_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -151.5298309, 432.6177063, -126.9808121, 367.7620544, -519.2918701, 559.5985107
1: -107.2448425, 270.8350220, -90.4662857, 230.2761536, -337.5209961, 361.3013000
2: -117.4931641, 249.1458130, -99.2604752, 212.6578979, -330.1510620, 348.4062805
3: -105.9395828, 325.0317383, -89.4693222, 275.6247559, -381.5643311, 414.5010071
4: -168.2697754, 265.7545471, -143.0995178, 225.5220642, -393.7917786, 408.8540039

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

## BFS NS instance: NS_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -151.5298309, 432.6177063, -147.2108154, 420.2660217, -571.7958374, 579.8284912
1: -107.2448425, 270.8350220, -104.2389603, 263.0170593, -370.2619019, 375.0739746
2: -117.4931641, 249.1458130, -114.2639771, 241.8386230, -359.3317871, 363.4097595
3: -105.9395828, 325.0317383, -102.9918060, 315.8397827, -421.7793274, 428.0235596
4: -168.2697754, 265.7545471, -163.5919800, 258.1581726, -426.4279175, 429.3465271

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

## BFS NS instance: NS_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -197.3084412, 564.2221680, -126.9808121, 367.7620544, -565.0704956, 690.4454956
1: -140.3680878, 353.6281128, -90.4662857, 230.2761536, -370.3646240, 444.0943909
2: -153.7138672, 325.5948486, -99.2604752, 212.6578979, -366.1944885, 424.8553162
3: -138.4787903, 424.5709534, -89.4693222, 275.6247559, -414.1035461, 513.7514038
4: -219.6260223, 347.3887634, -143.0995178, 225.5220642, -444.9217834, 490.4882507

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

## BFS NS instance: NS_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -197.3084412, 564.2221680, -147.2108154, 420.2660217, -617.2528687, 710.2968140
1: -140.3680878, 353.6281128, -104.2389603, 263.0170593, -402.9533386, 457.8670654
2: -153.7138672, 325.5948486, -114.2639771, 241.8386230, -395.2757263, 439.8587646
3: -138.4787903, 424.5709534, -102.9918060, 315.8397827, -454.2528992, 527.2028809
4: -219.6260223, 347.3887634, -163.5919800, 258.1581726, -477.3893738, 510.9073181

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 36

## BFS NS instance: NS_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -151.7067871, 432.6956787, -132.4024658, 382.8420715, -534.5488281, 565.0981445
1: -107.3257980, 271.0708008, -94.2508698, 239.8445587, -347.1702271, 365.3216553
2: -117.5263596, 249.3552856, -103.4209290, 221.5840149, -339.1103210, 352.7762146
3: -106.0172272, 325.1901550, -93.2020340, 286.9747314, -392.9919434, 418.3921814
4: -168.3002014, 265.9169922, -148.9710999, 234.8580780, -403.1582642, 414.8880310

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 2

## BFS NS instance: NS_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -151.7067871, 432.6956787, -151.9481201, 433.8209839, -585.5275269, 584.6437988
1: -107.3257980, 271.0708008, -107.5468903, 271.5707092, -378.8963623, 378.6176758
2: -117.5263596, 249.3552856, -117.8261566, 249.8620605, -367.3883972, 367.1814270
3: -106.0172272, 325.1901550, -106.2476196, 325.9197388, -431.9369202, 431.4377441
4: -168.3002014, 265.9169922, -168.7325439, 266.4584656, -434.7586060, 434.6495361

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

## BFS NS instance: NS_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -195.5906372, 559.0335083, -132.4024658, 382.8420715, -578.4327393, 690.7758789
1: -139.1471405, 350.4203796, -94.2508698, 239.8445587, -378.7415771, 444.6712341
2: -152.3687286, 322.5435791, -103.4209290, 221.5840149, -373.7952576, 425.9645081
3: -137.2671204, 420.7194519, -93.2020340, 286.9747314, -424.2418213, 513.6606445
4: -217.6697235, 344.2184143, -148.9710999, 234.8580780, -452.3428650, 493.1894836

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 23

## BFS NS instance: NS_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -195.5906372, 559.0335083, -151.9481201, 433.8209839, -629.2320557, 709.9721680
1: -139.1471405, 350.4203796, -107.5468903, 271.5707092, -410.3114929, 457.9672546
2: -152.3687286, 322.5435791, -117.8261566, 249.8620605, -401.9885559, 440.3697205
3: -137.2671204, 420.7194519, -106.2476196, 325.9197388, -463.1591797, 526.6428223
4: -217.6697235, 344.2184143, -168.7325439, 266.4584656, -483.7868652, 512.9498291

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.81 + 163.16 = 166.97 seconds
