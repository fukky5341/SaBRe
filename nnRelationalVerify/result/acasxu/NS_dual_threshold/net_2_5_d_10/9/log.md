## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 317.9962633056


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-147.6918945, 296.2000732, -147.6918945, 296.2000732, -443.8919678, 443.8919678)
1: (-54.2157822, 114.7664490, -54.2157822, 114.7664490, -168.9822083, 168.9822083)
2: (-26.4642086, 118.5023804, -26.4642086, 118.5023804, -144.9665833, 144.9665833)
3: (-64.3750992, 135.0477753, -64.3750992, 135.0477753, -199.4228668, 199.4228668)
4: (-33.7146263, 118.1404877, -33.7146263, 118.1404877, -151.8550873, 151.8550873)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.17 + 1.73 = 3.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -320.5607493, upper bound: 320.5607493

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.7722790, upper bound: 318.4649110
time: 0.57 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.4091271, upper bound: 318.4091271
time: 0.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.29 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.29
Output dim: 0, lower bound: -319.7722790, upper bound: 318.4649110
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.29
Output dim: 0, lower bound: -318.4091271, upper bound: 318.4091271

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -147.6918945, 296.2000732, -147.0832672, 295.0966797, -442.7885742, 443.2833252
1: -54.2157822, 114.7664490, -54.0130196, 114.3277054, -168.5434570, 168.7794342
2: -26.4642086, 118.5023804, -26.3555050, 118.0578232, -144.5220184, 144.8578796
3: -64.3750992, 135.0477753, -64.1222229, 134.5280457, -198.9031372, 199.1699982
4: -33.7146263, 118.1404877, -33.5792999, 117.6868362, -151.4014282, 151.7197571

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.4221397, upper bound: 318.1914766
time: 0.57 seconds

## Relational analysis of NS_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.2564185, upper bound: 318.1800438
time: 0.53 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.1010653, upper bound: 317.8635746
time: 0.64 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -144.5029449, 290.6383362, -149.9054565, 297.9695435, -442.4724731, 440.5437622
1: -53.1862755, 112.4669647, -55.5535355, 119.9638672, -173.1501160, 168.0205078
2: -25.8824501, 116.3153229, -26.6980762, 121.6315079, -147.5139618, 143.0133972
3: -63.0503235, 132.3593140, -66.7481308, 138.6580811, -201.7084045, 199.1074219
4: -32.9795837, 115.8836899, -34.3718414, 119.9604568, -152.9400177, 150.2555084

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.4091271, upper bound: 318.4091271
time: 0.62 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.4091271, upper bound: 318.4091271
time: 0.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.18 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 4.18
Output dim: 0, lower bound: -319.2564185, upper bound: 318.1800438
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 4.18
Output dim: 0, lower bound: -319.1010653, upper bound: 317.8635746
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 4.18
Output dim: 0, lower bound: -318.4091271, upper bound: 318.4091271
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 4.18
Output dim: 0, lower bound: -318.4091271, upper bound: 318.4091271

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -147.3860321, 295.5582581, -147.0832672, 295.0966797, -442.4826660, 442.6415405
1: -54.0963974, 114.5242462, -54.0130196, 114.3277054, -168.4240417, 168.5372620
2: -26.4097996, 118.2341003, -26.3555050, 118.0578232, -144.4676208, 144.5895844
3: -64.2462692, 134.7600403, -64.1222229, 134.5280457, -198.7742920, 198.8822632
4: -33.6457100, 117.8788071, -33.5792999, 117.6868362, -151.3325043, 151.4580841

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.1010653, upper bound: 317.8635746
time: 0.66 seconds

## Relational analysis of NS_B1_A1_B2

### Relational analysis result of NS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.1010653, upper bound: 317.8635746
time: 0.66 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -152.4238892, 303.2209778, -146.2759705, 293.6509399, -446.0747986, 449.4969177
1: -55.4628716, 117.6857910, -53.7371712, 113.6949463, -169.1578217, 171.4229431
2: -27.2649250, 121.0455322, -26.2074909, 117.4631577, -144.7280884, 147.2530212
3: -66.1778564, 138.2198639, -63.7710991, 133.7972870, -199.9751282, 201.9909668
4: -34.6924782, 120.8341064, -33.3877869, 117.0877991, -151.7802277, 154.2218933

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A2_A1

### Relational analysis result of NS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.1010653, upper bound: 317.8635746
time: 0.60 seconds

## Relational analysis of NS_B1_A2_A2

### Relational analysis result of NS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.1010653, upper bound: 317.8635746
time: 0.64 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -147.0814056, 295.0932617, -149.9054565, 297.9695435, -445.0509644, 444.9987183
1: -54.0123825, 114.3264313, -55.5535355, 119.9638672, -173.9762573, 169.8799744
2: -26.3551407, 118.0565643, -26.6980762, 121.6315079, -147.9866333, 144.7546387
3: -64.1215057, 134.5265045, -66.7481308, 138.6580811, -202.7795868, 201.2746277
4: -33.5788727, 117.6855392, -34.3718414, 119.9604568, -153.5393066, 152.0573730

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.3303007, upper bound: 318.3682215
time: 0.65 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.3284959, upper bound: 318.3284959
time: 0.57 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -149.0563965, 296.4791565, -149.9054565, 297.9695435, -447.0259399, 446.3846130
1: -55.2723312, 119.4071503, -55.5535355, 119.9638672, -175.2361450, 174.9606934
2: -26.5379124, 121.0739441, -26.6980762, 121.6315079, -148.1694183, 147.7720032
3: -66.4325180, 137.9850922, -66.7481308, 138.6580811, -205.0905762, 204.7332153
4: -34.1861267, 119.3899536, -34.3718414, 119.9604568, -154.1465759, 153.7617798

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.0526793, upper bound: 317.8210533
time: 0.67 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -317.8113604, upper bound: 317.8113604
time: 0.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.40 seconds
NS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 6.40
Output dim: 0, lower bound: -319.1010653, upper bound: 317.8635746
NS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 6.40
Output dim: 0, lower bound: -319.1010653, upper bound: 317.8635746
NS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 6.40
Output dim: 0, lower bound: -319.1010653, upper bound: 317.8635746
NS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 6.40
Output dim: 0, lower bound: -319.1010653, upper bound: 317.8635746
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 6.40
Output dim: 0, lower bound: -318.3303007, upper bound: 318.3682215
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 6.40
Output dim: 0, lower bound: -318.3284959, upper bound: 318.3284959
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 6.40
Output dim: 0, lower bound: -318.0526793, upper bound: 317.8210533
NS_B2_A2_B2, status: Status.VERIFIED, split count: 3, time: 6.40
Output dim: 0, lower bound: -317.8113604, upper bound: 317.8113604

## BFS NS instance: NS_B1_A1_B1

### Backsubstitution after applying NS history:
0: -147.3860321, 295.5582581, -146.7791748, 294.4573975, -441.8433228, 442.3374329
1: -54.0963974, 114.5242462, -53.8941994, 114.0864563, -168.1828156, 168.4184418
2: -26.4097996, 118.2341003, -26.3014393, 117.7901535, -144.1999054, 144.5355377
3: -64.2462692, 134.7600403, -63.9930229, 134.2416992, -198.4879303, 198.7530670
4: -33.6457100, 117.8788071, -33.5107918, 117.4259338, -151.0716400, 151.3896027

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.2564185, upper bound: 318.1800438
time: 0.63 seconds

## Relational analysis of NS_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.2564185, upper bound: 318.1800438
time: 0.65 seconds

## BFS NS instance: NS_B1_A1_B2

### Backsubstitution after applying NS history:
0: -147.3860321, 295.5582581, -151.8265839, 302.1383972, -449.5243835, 447.3848267
1: -54.0963974, 114.5242462, -55.2644691, 117.2561798, -171.3525543, 169.7887115
2: -26.4097996, 118.2341003, -27.1587963, 120.6082306, -147.0180206, 145.3928986
3: -64.2462692, 134.7600403, -65.9293518, 137.7100677, -201.9563141, 200.6893616
4: -33.6457100, 117.8788071, -34.5604095, 120.3883972, -154.0341034, 152.4392090

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_B2_B1

### Relational analysis result of NS_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.2379960, upper bound: 318.1691993
time: 0.63 seconds

## Relational analysis of NS_B1_A1_B2_B2

### Relational analysis result of NS_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.5705160, upper bound: 318.0653022
time: 0.59 seconds

## BFS NS instance: NS_B1_A2_A1

### Backsubstitution after applying NS history:
0: -151.8265839, 302.1383972, -146.2759705, 293.6509399, -445.4775085, 448.4143372
1: -55.2644691, 117.2561798, -53.7371712, 113.6949463, -168.9594116, 170.9933472
2: -27.1587963, 120.6082306, -26.2074909, 117.4631577, -144.6219482, 146.8157043
3: -65.9293518, 137.7100677, -63.7710991, 133.7972870, -199.7266235, 201.4811707
4: -34.5604095, 120.3883972, -33.3877869, 117.0877991, -151.6481934, 153.7761841

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.1010653, upper bound: 317.8635746
time: 0.65 seconds

## Relational analysis of NS_B1_A2_A1_B2

### Relational analysis result of NS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.1010653, upper bound: 317.8635746
time: 0.62 seconds

## BFS NS instance: NS_B1_A2_A2

### Backsubstitution after applying NS history:
0: -154.0044098, 303.5567627, -146.2759705, 293.6509399, -447.6553345, 449.8327026
1: -56.5648041, 122.2215118, -53.7371712, 113.6949463, -170.2597504, 175.9586792
2: -27.3452854, 123.5617981, -26.2074909, 117.4631577, -144.8084106, 149.7692871
3: -67.9508591, 141.2708282, -63.7710991, 133.7972870, -201.7481232, 205.0419312
4: -35.1232300, 121.9159317, -33.3877869, 117.0877991, -152.2110138, 155.3037109

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.0804904, upper bound: 317.8538306
time: 0.67 seconds

## Relational analysis of NS_B1_A2_A2_B2

### Relational analysis result of NS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.5607706, upper bound: 317.8323165
time: 0.64 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -141.6918793, 285.8075562, -148.9473724, 296.3515930, -438.0434570, 434.7549438
1: -52.2652931, 110.4355850, -55.2437668, 119.2898331, -171.5551147, 165.6793518
2: -25.3971939, 114.3052826, -26.5306320, 120.9745636, -146.3717651, 140.8359070
3: -61.9307518, 130.0001526, -66.3735275, 137.8640442, -199.7947998, 196.3736572
4: -32.3711700, 113.8847809, -34.1625175, 119.3026733, -151.6738434, 148.0472565

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A1_A1

### Relational analysis result of NS_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.3677914, upper bound: 319.7296873
time: 0.64 seconds

## Relational analysis of NS_B2_A1_A1_A2

### Relational analysis result of NS_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.3232875, upper bound: 318.6426534
time: 0.57 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -146.6456604, 294.0655518, -149.2581024, 296.8665466, -443.5122070, 443.3236389
1: -53.8395729, 114.0519257, -55.3403244, 119.4870605, -173.3265991, 169.3922272
2: -26.2638226, 117.6881638, -26.5823040, 121.1782303, -147.4420471, 144.2704620
3: -63.9587898, 134.1163483, -66.4853516, 138.0952148, -202.0540009, 200.6016998
4: -33.4695091, 117.3049545, -34.2247467, 119.5053482, -152.9748535, 151.5296936

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A1_A2_A1

### Relational analysis result of NS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -317.8765404, upper bound: 319.4235998
time: 0.64 seconds

## Relational analysis of NS_B2_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A2_A1

### Relational analysis result of NS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.3667522, upper bound: 319.7126212
time: 0.63 seconds

## Relational analysis of NS_B2_A1_A2_A2

### Relational analysis result of NS_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.3268360, upper bound: 319.0040778
time: 0.60 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -149.0563965, 296.4791565, -149.6340179, 297.3988342, -446.4551697, 446.1131592
1: -55.2723312, 119.4071503, -55.4502029, 119.7550278, -175.0273285, 174.8573608
2: -26.5379124, 121.0739441, -26.6500664, 121.4149933, -147.9529114, 147.7239685
3: -66.4325180, 137.9850922, -66.6370010, 138.4062195, -204.8387451, 204.6221008
4: -34.1861267, 119.3899536, -34.3101044, 119.7490845, -153.9352112, 153.7000427

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -317.8113604, upper bound: 317.8113604
time: 0.64 seconds

## Relational analysis of NS_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -317.8113604, upper bound: 317.8113604
time: 0.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.27 seconds
NS_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -319.2564185, upper bound: 318.1800438
NS_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -319.2564185, upper bound: 318.1800438
NS_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -319.2379960, upper bound: 318.1691993
NS_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -318.5705160, upper bound: 318.0653022
NS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -319.1010653, upper bound: 317.8635746
NS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -319.1010653, upper bound: 317.8635746
NS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -319.0804904, upper bound: 317.8538306
NS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -318.5607706, upper bound: 317.8323165
NS_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -318.3677914, upper bound: 319.7296873
NS_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -318.3232875, upper bound: 318.6426534
NS_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -318.3667522, upper bound: 319.7126212
NS_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -318.3268360, upper bound: 319.0040778
NS_B2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 4.27
Output dim: 0, lower bound: -317.8113604, upper bound: 317.8113604
NS_B2_A2_B1_A2, status: Status.VERIFIED, split count: 4, time: 4.27
Output dim: 0, lower bound: -317.8113604, upper bound: 317.8113604

## BFS NS instance: NS_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -146.7791748, 294.4573975, -146.7791748, 294.4573975, -441.2365723, 441.2365723
1: -53.8941994, 114.0864563, -53.8941994, 114.0864563, -167.9806519, 167.9806519
2: -26.3014393, 117.7901535, -26.3014393, 117.7901535, -144.0915527, 144.0915527
3: -63.9930229, 134.2416992, -63.9930229, 134.2416992, -198.2347260, 198.2347260
4: -33.5107918, 117.4259338, -33.5107918, 117.4259338, -150.9367218, 150.9367218

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_B1_A1_B1

### Relational analysis result of NS_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.7553449, upper bound: 318.4452314
time: 0.69 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2

### Relational analysis result of NS_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.0053465, upper bound: 318.4030565
time: 0.60 seconds

## BFS NS instance: NS_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -148.7932129, 295.9226990, -146.7791748, 294.4573975, -443.2505493, 442.7018738
1: -55.1717758, 119.2037811, -53.8941994, 114.0864563, -169.2582092, 173.0979462
2: -26.4914856, 120.8627625, -26.3014393, 117.7901535, -144.2815857, 147.1641998
3: -66.3245239, 137.7397766, -63.9930229, 134.2416992, -200.5662231, 201.7328033
4: -34.1261711, 119.1840820, -33.5107918, 117.4259338, -151.5521088, 152.6948700

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_B1_A2_B1

### Relational analysis result of NS_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.7553449, upper bound: 318.4452314
time: 0.62 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2

### Relational analysis result of NS_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.0053465, upper bound: 318.4030565
time: 0.60 seconds

## BFS NS instance: NS_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -147.3860321, 295.5582581, -151.1171875, 300.9364929, -448.3224487, 446.6754150
1: -54.0963974, 114.5242462, -55.0382729, 116.7512741, -170.8476410, 169.5625153
2: -26.4097996, 118.2341003, -27.0332279, 120.1250458, -146.5348358, 145.2673340
3: -64.2462692, 134.7600403, -65.6367645, 137.1298981, -201.3761444, 200.3967590
4: -33.6457100, 117.8788071, -34.4027023, 119.8971176, -153.5428314, 152.2815094

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_B2_B1_A1

### Relational analysis result of NS_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.2379960, upper bound: 318.1691993
time: 0.66 seconds

## Relational analysis of NS_B1_A1_B2_B1_A2

### Relational analysis result of NS_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.2379960, upper bound: 318.1691993
time: 0.67 seconds

## BFS NS instance: NS_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -144.9864807, 291.6065979, -134.7574463, 275.9481506, -420.9346313, 426.3640442
1: -53.3450203, 112.8114929, -50.3087044, 105.9661560, -159.3111725, 163.1201935
2: -25.9868755, 116.6689606, -24.1924667, 110.6458969, -136.6327515, 140.8613892
3: -63.2793884, 132.8180084, -59.3704910, 124.7822495, -188.0615997, 192.1885071
4: -33.1050720, 116.2900848, -30.7387390, 110.2618256, -143.3668976, 147.0288239

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_B1_A1_B2_B2_A1

### Relational analysis result of NS_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.5705160, upper bound: 318.0653022
time: 0.66 seconds

## Relational analysis of NS_B1_A1_B2_B2_A2

### Relational analysis result of NS_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.5705160, upper bound: 318.0653022
time: 0.67 seconds

## BFS NS instance: NS_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -151.8265839, 302.1383972, -146.7791748, 294.4573975, -446.2839050, 448.9175720
1: -55.2644691, 117.2561798, -53.8941994, 114.0864563, -169.3509216, 171.1503601
2: -27.1587963, 120.6082306, -26.3014393, 117.7901535, -144.9489136, 146.9096527
3: -65.9293518, 137.7100677, -63.9930229, 134.2416992, -200.1710510, 201.7030945
4: -34.5604095, 120.3883972, -33.5107918, 117.4259338, -151.9863434, 153.8991852

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_A1_B1_A1

### Relational analysis result of NS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.6385663, upper bound: 319.7047996
time: 0.60 seconds

## Relational analysis of NS_B1_A2_A1_B1_A2

### Relational analysis result of NS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.5909889, upper bound: 318.5909889
time: 0.71 seconds

## BFS NS instance: NS_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -151.8265839, 302.1383972, -151.5626068, 301.6701965, -453.4967346, 453.7009888
1: -55.2644691, 117.2561798, -55.1760941, 117.0798569, -172.3443298, 172.4322815
2: -27.1587963, 120.6082306, -27.1079063, 120.4307861, -147.5895538, 147.7161102
3: -65.9293518, 137.7100677, -65.8264160, 137.4945831, -203.4239349, 203.5364838
4: -34.5604095, 120.3883972, -34.5013924, 120.2073898, -154.7677917, 154.8897858

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_A1_B2_A1

### Relational analysis result of NS_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.6385663, upper bound: 319.7047996
time: 0.63 seconds

## Relational analysis of NS_B1_A2_A1_B2_A2

### Relational analysis result of NS_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.5909889, upper bound: 318.5909889
time: 0.61 seconds

## BFS NS instance: NS_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -154.0044098, 303.5567627, -145.5251007, 292.3727722, -446.3771973, 449.0818481
1: -56.5648041, 122.2215118, -53.4960327, 113.1577454, -169.7225494, 175.7175446
2: -27.3452854, 123.5617981, -26.0739193, 116.9523010, -144.2975311, 149.6357117
3: -67.9508591, 141.2708282, -63.4645767, 133.1797638, -201.1305847, 204.7353668
4: -35.1232300, 121.9159317, -33.2197952, 116.5663528, -151.6895752, 155.1357269

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_A2_A2_B1_B1

### Relational analysis result of NS_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.0804904, upper bound: 317.8538306
time: 0.70 seconds

## Relational analysis of NS_B1_A2_A2_B1_B2

### Relational analysis result of NS_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.0804904, upper bound: 317.8538306
time: 0.67 seconds

## BFS NS instance: NS_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -151.8799896, 300.0800476, -129.8098450, 268.1188965, -419.9989014, 429.8898010
1: -55.9011116, 120.7074127, -48.9168282, 103.1911087, -159.0921783, 169.6242371
2: -26.9719791, 122.1550369, -23.3331871, 107.8744507, -134.8464355, 145.4881897
3: -67.1147690, 139.4687653, -57.5079155, 121.2722015, -188.3869629, 196.9766846
4: -34.6472626, 120.4939041, -29.7183933, 107.1924973, -141.8397522, 150.2122803

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_B1_A2_A2_B2_B1

### Relational analysis result of NS_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.5607706, upper bound: 317.8323165
time: 0.67 seconds

## Relational analysis of NS_B1_A2_A2_B2_B2

### Relational analysis result of NS_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.5607706, upper bound: 317.8323165
time: 0.64 seconds

## BFS NS instance: NS_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -140.9439850, 284.5309753, -148.9473724, 296.3515930, -437.2955933, 433.4783325
1: -52.0243225, 109.8980255, -55.2437668, 119.2898331, -171.3141174, 165.1417847
2: -25.2635994, 113.7951508, -26.5306320, 120.9745636, -146.2381439, 140.3257751
3: -61.6253319, 129.3824921, -66.3735275, 137.8640442, -199.4893799, 195.7560120
4: -32.2032166, 113.3641968, -34.1625175, 119.3026733, -151.5058899, 147.5267029

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.3677914, upper bound: 319.7296873
time: 0.60 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.3677914, upper bound: 319.7296873
time: 0.64 seconds

## BFS NS instance: NS_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -125.5645752, 261.0166626, -146.8484802, 292.8802490, -418.4447937, 407.8651123
1: -47.5888519, 100.3641891, -54.5793877, 117.7769928, -165.3658447, 154.9435730
2: -22.5948162, 105.0696259, -26.1553574, 119.5726547, -142.1674500, 131.2249756
3: -55.8606834, 117.8552094, -65.5268402, 136.0497437, -191.9104156, 183.3820496
4: -28.8149567, 104.3087234, -33.6830635, 117.8793640, -146.6943054, 137.9917755

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A1_A2_B1

### Relational analysis result of NS_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.3232875, upper bound: 318.6426534
time: 0.61 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2

### Relational analysis result of NS_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.3232875, upper bound: 318.6426534
time: 0.62 seconds

## BFS NS instance: NS_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -145.9067841, 292.8096619, -149.2581024, 296.8665466, -442.7733154, 442.0676880
1: -53.6023979, 113.5242996, -55.3403244, 119.4870605, -173.0894318, 168.8645782
2: -26.1331673, 117.1856461, -26.5823040, 121.1782303, -147.3114014, 143.7679443
3: -63.6583138, 133.5087128, -66.4853516, 138.0952148, -201.7535248, 199.9940643
4: -33.3043480, 116.7929077, -34.2247467, 119.5053482, -152.8096924, 151.0176239

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A2_A1_B1

### Relational analysis result of NS_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.3667522, upper bound: 319.7126212
time: 0.74 seconds

## Relational analysis of NS_B2_A1_A2_A1_B2

### Relational analysis result of NS_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.3667522, upper bound: 319.7126212
time: 0.66 seconds

## BFS NS instance: NS_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -129.9613342, 268.2101440, -147.1764679, 293.4196472, -423.3809814, 415.3865967
1: -48.9637794, 103.4860077, -54.6808777, 117.9848785, -166.9486542, 158.1668549
2: -23.3542976, 108.0833130, -26.2097435, 119.7861557, -143.1404572, 134.2930603
3: -57.5968552, 121.4740753, -65.6448822, 136.2930908, -193.8899536, 187.1189575
4: -29.7535763, 107.3545532, -33.7488098, 118.0916595, -147.8452301, 141.1033630

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A2_A2_B1

### Relational analysis result of NS_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.3268360, upper bound: 318.7455165
time: 0.81 seconds

## Relational analysis of NS_B2_A1_A2_A2_B2

### Relational analysis result of NS_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.3268360, upper bound: 319.0040784
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.70 seconds
NS_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -319.7553449, upper bound: 318.4452314
NS_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -319.0053465, upper bound: 318.4030565
NS_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -319.7553449, upper bound: 318.4452314
NS_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -319.0053465, upper bound: 318.4030565
NS_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -319.2379960, upper bound: 318.1691993
NS_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -319.2379960, upper bound: 318.1691993
NS_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -318.5705160, upper bound: 318.0653022
NS_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -318.5705160, upper bound: 318.0653022
NS_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -318.6385663, upper bound: 319.7047996
NS_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -318.5909889, upper bound: 318.5909889
NS_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -318.6385663, upper bound: 319.7047996
NS_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -318.5909889, upper bound: 318.5909889
NS_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -319.0804904, upper bound: 317.8538306
NS_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -319.0804904, upper bound: 317.8538306
NS_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -318.5607706, upper bound: 317.8323165
NS_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -318.5607706, upper bound: 317.8323165
NS_B2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -318.3677914, upper bound: 319.7296873
NS_B2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -318.3677914, upper bound: 319.7296873
NS_B2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -318.3232875, upper bound: 318.6426534
NS_B2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -318.3232875, upper bound: 318.6426534
NS_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -318.3667522, upper bound: 319.7126212
NS_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -318.3667522, upper bound: 319.7126212
NS_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -318.3268360, upper bound: 318.7455165
NS_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -318.3268360, upper bound: 319.0040784

## BFS NS instance: NS_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -146.7791748, 294.4573975, -146.0284119, 293.1794434, -439.9586182, 440.4856873
1: -53.8941994, 114.0864563, -53.6530647, 113.5495758, -167.4437714, 167.7395020
2: -26.3014393, 117.7901535, -26.1678429, 117.2793808, -143.5808105, 143.9579773
3: -63.9930229, 134.2416992, -63.6867294, 133.6244202, -197.6174469, 197.9284363
4: -33.5107918, 117.4259338, -33.3424187, 116.9046783, -150.4154663, 150.7683563

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.0441021, upper bound: 319.0441021
time: 0.63 seconds

## Relational analysis of NS_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.0441021, upper bound: 319.0441021
time: 0.64 seconds

## BFS NS instance: NS_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -144.3811035, 290.5074768, -130.1889343, 268.6842041, -413.0653076, 420.6964111
1: -53.1431618, 112.3736801, -49.0288544, 103.4790955, -156.6222534, 161.4025269
2: -25.8786430, 116.2254333, -23.4022141, 108.1495972, -134.0282135, 139.6276245
3: -63.0264778, 132.3000946, -57.6666183, 121.6073227, -184.6337585, 189.9667053
4: -32.9703102, 115.8377914, -29.8077126, 107.4550095, -140.4252777, 145.6455078

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.0441021, upper bound: 319.0441021
time: 0.69 seconds

## Relational analysis of NS_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.0441021, upper bound: 319.0441021
time: 0.66 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -148.7932129, 295.9226990, -146.0284119, 293.1794434, -441.9726257, 441.9511108
1: -55.1717758, 119.2037811, -53.6530647, 113.5495758, -168.7213440, 172.8568115
2: -26.4914856, 120.8627625, -26.1678429, 117.2793808, -143.7708435, 147.0306091
3: -66.3245239, 137.7397766, -63.6867294, 133.6244202, -199.9489288, 201.4265137
4: -34.1261711, 119.1840820, -33.3424187, 116.9046783, -151.0308533, 152.5264740

Time for backsubstitution: 3.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.9785179, upper bound: 318.2963916
time: 0.73 seconds

## Relational analysis of NS_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.9785179, upper bound: 318.4030565
time: 0.72 seconds

## BFS NS instance: NS_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -146.7100220, 292.4746704, -130.1889343, 268.6842041, -415.3942261, 422.6636047
1: -54.5122223, 117.7009354, -49.0288544, 103.4790955, -157.9913177, 166.7297974
2: -26.1187038, 119.4696274, -23.4022141, 108.1495972, -134.2682953, 142.8718414
3: -65.4839325, 135.9376068, -57.6666183, 121.6073227, -187.0912170, 193.6042175
4: -33.6499443, 117.7696609, -29.8077126, 107.4550095, -141.1049500, 147.5773621

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.9785179, upper bound: 318.2963916
time: 0.66 seconds

## Relational analysis of NS_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.9785179, upper bound: 318.4030563
time: 0.79 seconds

## BFS NS instance: NS_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -146.7791748, 294.4573975, -151.1171875, 300.9364929, -447.7156677, 445.5744934
1: -53.8941994, 114.0864563, -55.0382729, 116.7512741, -170.6454620, 169.1247253
2: -26.3014393, 117.7901535, -27.0332279, 120.1250458, -146.4264679, 144.8233490
3: -63.9930229, 134.2416992, -65.6367645, 137.1298981, -201.1229248, 199.8784485
4: -33.5107918, 117.4259338, -34.4027023, 119.8971176, -153.4079132, 151.8286438

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_B2_B1_A1_A1

### Relational analysis result of NS_B1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.9558761, upper bound: 317.8847282
time: 0.75 seconds

## Relational analysis of NS_B1_A1_B2_B1_A1_A2

### Relational analysis result of NS_B1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.9558761, upper bound: 318.1691993
time: 0.74 seconds

## BFS NS instance: NS_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -148.7932129, 295.9226990, -151.1171875, 300.9364929, -449.7296448, 447.0398560
1: -55.1717758, 119.2037811, -55.0382729, 116.7512741, -171.9230194, 174.2420502
2: -26.4914856, 120.8627625, -27.0332279, 120.1250458, -146.6165009, 147.8959961
3: -66.3245239, 137.7397766, -65.6367645, 137.1298981, -203.4544067, 203.3765106
4: -34.1261711, 119.1840820, -34.4027023, 119.8971176, -154.0232849, 153.5867920

Time for backsubstitution: 2.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_B2_B1_A2_A1

### Relational analysis result of NS_B1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.9558761, upper bound: 317.8847282
time: 0.79 seconds

## Relational analysis of NS_B1_A1_B2_B1_A2_A2

### Relational analysis result of NS_B1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.9558761, upper bound: 318.1691993
time: 0.73 seconds

## BFS NS instance: NS_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -144.3811035, 290.5074768, -134.7574463, 275.9481506, -420.3292236, 425.2648926
1: -53.1431618, 112.3736801, -50.3087044, 105.9661560, -159.1093140, 162.6823883
2: -25.8786430, 116.2254333, -24.1924667, 110.6458969, -136.5245209, 140.4178772
3: -63.0264778, 132.3000946, -59.3704910, 124.7822495, -187.8087158, 191.6705627
4: -32.9703102, 115.8377914, -30.7387390, 110.2618256, -143.2321167, 146.5765381

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_B2_B2_A1_A1

### Relational analysis result of NS_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.5549443, upper bound: 317.6893191
time: 0.75 seconds

## Relational analysis of NS_B1_A1_B2_B2_A1_A2

### Relational analysis result of NS_B1_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.5549443, upper bound: 318.0653022
time: 0.69 seconds

## BFS NS instance: NS_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -146.7100220, 292.4746704, -134.7574463, 275.9481506, -422.6581726, 427.2321167
1: -54.5122223, 117.7009354, -50.3087044, 105.9661560, -160.4783783, 168.0096436
2: -26.1187038, 119.4696274, -24.1924667, 110.6458969, -136.7645874, 143.6620941
3: -65.4839325, 135.9376068, -59.3704910, 124.7822495, -190.2661285, 195.3081055
4: -33.6499443, 117.7696609, -30.7387390, 110.2618256, -143.9117737, 148.5083923

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A1_B2_B2_A2_A1

### Relational analysis result of NS_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.5549443, upper bound: 317.6893191
time: 0.77 seconds

## Relational analysis of NS_B1_A1_B2_B2_A2_A2

### Relational analysis result of NS_B1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.5549443, upper bound: 318.0653022
time: 0.85 seconds

## BFS NS instance: NS_B1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -151.1171875, 300.9364929, -146.7791748, 294.4573975, -445.5745239, 447.7156677
1: -55.0382729, 116.7512741, -53.8941994, 114.0864563, -169.1247253, 170.6454620
2: -27.0332279, 120.1250458, -26.3014393, 117.7901535, -144.8233490, 146.4264679
3: -65.6367645, 137.1298981, -63.9930229, 134.2416992, -199.8784485, 201.1229248
4: -34.4027023, 119.8971176, -33.5107918, 117.4259338, -151.8286438, 153.4079132

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_A1_B1_A1_B1

### Relational analysis result of NS_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.0258194, upper bound: 318.6092716
time: 0.65 seconds

## Relational analysis of NS_B1_A2_A1_B1_A1_B2

### Relational analysis result of NS_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.0258194, upper bound: 318.6092716
time: 0.69 seconds

## BFS NS instance: NS_B1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -134.7574463, 275.9481506, -144.3811035, 290.5074768, -425.2648621, 420.3292236
1: -50.3087044, 105.9661560, -53.1431618, 112.3736801, -162.6823883, 159.1093140
2: -24.1924667, 110.6458969, -25.8786430, 116.2254333, -140.4178772, 136.5245209
3: -59.3704910, 124.7822495, -63.0264778, 132.3000946, -191.6705627, 187.8087158
4: -30.7387390, 110.2618256, -32.9703102, 115.8377914, -146.5765381, 143.2321014

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_A1_B1_A2_B1

### Relational analysis result of NS_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.0258194, upper bound: 318.6092716
time: 0.77 seconds

## Relational analysis of NS_B1_A2_A1_B1_A2_B2

### Relational analysis result of NS_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -319.0258194, upper bound: 318.6092716
time: 0.70 seconds

## BFS NS instance: NS_B1_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -151.1171875, 300.9364929, -151.5626068, 301.6701965, -452.7873535, 452.4990845
1: -55.0382729, 116.7512741, -55.1760941, 117.0798569, -172.1181335, 171.9273682
2: -27.0332279, 120.1250458, -27.1079063, 120.4307861, -147.4640045, 147.2329102
3: -65.6367645, 137.1298981, -65.8264160, 137.4945831, -203.1313171, 202.9563141
4: -34.4027023, 119.8971176, -34.5013924, 120.2073898, -154.6100922, 154.3984985

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_A1_B2_A1_B1

### Relational analysis result of NS_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.5909889, upper bound: 318.5909889
time: 0.65 seconds

## Relational analysis of NS_B1_A2_A1_B2_A1_B2

### Relational analysis result of NS_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.5909889, upper bound: 318.5909889
time: 0.67 seconds

## BFS NS instance: NS_B1_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -134.7574463, 275.9481506, -149.2130432, 297.8479004, -432.6053467, 425.1611938
1: -50.3087044, 105.9661560, -54.4518013, 115.4236526, -165.7323456, 160.4179535
2: -24.1924667, 110.6458969, -26.6993103, 118.9061508, -143.0986176, 137.3452148
3: -59.3704910, 124.7822495, -64.8760910, 135.6221008, -194.9925842, 189.6583099
4: -30.7387390, 110.2618256, -33.9798813, 118.6714478, -149.4101562, 144.2416992

Time for backsubstitution: 2.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_A1_B2_A2_B1

### Relational analysis result of NS_B1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.5909889, upper bound: 318.5909889
time: 0.74 seconds

## Relational analysis of NS_B1_A2_A1_B2_A2_B2

### Relational analysis result of NS_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.5909889, upper bound: 318.5909889
time: 0.77 seconds

## BFS NS instance: NS_B1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -154.0044098, 303.5567627, -146.0284119, 293.1794434, -447.1838379, 449.5851135
1: -56.5648041, 122.2215118, -53.6530647, 113.5495758, -170.1143799, 175.8745575
2: -27.3452854, 123.5617981, -26.1678429, 117.2793808, -144.6246185, 149.7296448
3: -67.9508591, 141.2708282, -63.6867294, 133.6244202, -201.5752411, 204.9575500
4: -35.1232300, 121.9159317, -33.3424187, 116.9046783, -152.0279083, 155.2583466

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## BFS NS instance: NS_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -154.0044098, 303.5567627, -150.8536224, 300.4692383, -454.4736328, 454.4103699
1: -56.5648041, 122.2215118, -54.9500847, 116.5752869, -173.1400909, 177.1715698
2: -27.3452854, 123.5617981, -26.9824333, 119.9479828, -147.2932434, 150.5442352
3: -67.9508591, 141.2708282, -65.5340042, 136.9148254, -204.8656311, 206.8048096
4: -35.1232300, 121.9159317, -34.3438072, 119.7164993, -154.8397217, 156.2597351

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## BFS NS instance: NS_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -151.8799896, 300.0800476, -130.1865082, 268.6792908, -420.5592346, 430.2665405
1: -55.9011116, 120.7074127, -49.0279350, 103.4773560, -159.3784637, 169.7353516
2: -26.9719791, 122.1550369, -23.4017029, 108.1477585, -135.1197357, 145.5567169
3: -67.1147690, 139.4687653, -57.6653061, 121.6052551, -188.7200012, 197.1340637
4: -34.6472626, 120.4939041, -29.8071079, 107.4530411, -142.1002655, 150.3009949

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 34

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -151.8799896, 300.0800476, -134.5044556, 275.5016479, -427.3816528, 434.5845032
1: -55.9011116, 120.7074127, -50.2241440, 105.7980423, -161.6991577, 170.9315491
2: -26.9719791, 122.1550369, -24.1433125, 110.4786224, -137.4506073, 146.2983246
3: -67.1147690, 139.4687653, -59.2745934, 124.5759811, -191.6907501, 198.7433624
4: -34.6472626, 120.4939041, -30.6818123, 110.0898514, -144.7371216, 151.1756744

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_B2_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -140.9439850, 284.5309753, -146.6050415, 292.4125366, -433.3565063, 431.1360168
1: -52.0243225, 109.8980255, -54.4898300, 117.6473160, -169.6716003, 164.3878479
2: -25.2635994, 113.7951508, -26.1217022, 119.3767014, -144.6402893, 139.9168549
3: -61.6253319, 129.3824921, -65.4615479, 135.9299927, -197.5553284, 194.8439941
4: -32.2032166, 113.3641968, -33.6518402, 117.7018890, -149.9050903, 147.0160370

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A1_A1_B1_B1

### Relational analysis result of NS_B2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.2109227, upper bound: 319.6408668
time: 0.70 seconds

## Relational analysis of NS_B2_A1_A1_A1_B1_B2

### Relational analysis result of NS_B2_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.2109227, upper bound: 319.7296873
time: 0.90 seconds

## BFS NS instance: NS_B2_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -140.9439850, 284.5309753, -150.8104706, 299.3354492, -440.2794189, 435.3414001
1: -52.0243225, 109.8980255, -55.8370361, 120.7850342, -172.8093262, 165.7350464
2: -25.2635994, 113.7951508, -26.8651829, 122.3678818, -147.6314697, 140.6603088
3: -61.6253319, 129.3824921, -67.1717453, 139.5548706, -201.1801605, 196.5542297
4: -32.2032166, 113.3641968, -34.5897942, 120.6810532, -152.8842621, 147.9539642

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A1_A1_B2_B1

### Relational analysis result of NS_B2_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.2109227, upper bound: 319.6408668
time: 0.63 seconds

## Relational analysis of NS_B2_A1_A1_A1_B2_B2

### Relational analysis result of NS_B2_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.2109227, upper bound: 319.7296873
time: 0.75 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -125.5645752, 261.0166626, -144.5021515, 288.9411926, -414.5057678, 405.5187988
1: -47.5888519, 100.3641891, -53.8252144, 116.1332703, -163.7221222, 154.1894073
2: -22.5948162, 105.0696259, -25.7462177, 117.9775848, -140.5724030, 130.8158264
3: -55.8606834, 117.8552094, -64.6137085, 134.1132355, -189.9738922, 182.4689178
4: -28.8149567, 104.3087234, -33.1719093, 116.2824936, -145.0974426, 137.4806366

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A1_A2_B1_B1

### Relational analysis result of NS_B2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.1698742, upper bound: 318.6362369
time: 0.67 seconds

## Relational analysis of NS_B2_A1_A1_A2_B1_B2

### Relational analysis result of NS_B2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.1698742, upper bound: 318.6362369
time: 0.55 seconds

## BFS NS instance: NS_B2_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -125.5645752, 261.0166626, -148.6785736, 295.7914124, -421.3558960, 409.6952515
1: -47.5888519, 100.3641891, -55.1593781, 119.2441025, -166.8329468, 155.5235596
2: -22.5948162, 105.0696259, -26.4833279, 120.9312363, -143.5260315, 131.5529480
3: -55.8606834, 117.8552094, -66.3092117, 137.7049255, -193.5655975, 184.1644135
4: -28.8149567, 104.3087234, -34.1019554, 119.2238312, -148.0387878, 138.4106750

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A1_A2_B2_B1

### Relational analysis result of NS_B2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.1698742, upper bound: 318.6362369
time: 0.66 seconds

## Relational analysis of NS_B2_A1_A1_A2_B2_B2

### Relational analysis result of NS_B2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.1698742, upper bound: 318.6362369
time: 0.65 seconds

## BFS NS instance: NS_B2_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -145.9067841, 292.8096619, -146.6050415, 292.4125366, -438.3193359, 439.4146118
1: -53.6023979, 113.5242996, -54.4898300, 117.6473160, -171.2497101, 168.0140991
2: -26.1331673, 117.1856461, -26.1217022, 119.3767014, -145.5098724, 143.3073120
3: -63.6583138, 133.5087128, -65.4615479, 135.9299927, -199.5883026, 198.9702301
4: -33.3043480, 116.7929077, -33.6518402, 117.7018890, -151.0062103, 150.4447327

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A2_A1_B1_B1

### Relational analysis result of NS_B2_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.2098761, upper bound: 319.6242191
time: 0.77 seconds

## Relational analysis of NS_B2_A1_A2_A1_B1_B2

### Relational analysis result of NS_B2_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.2098761, upper bound: 319.7126212
time: 0.77 seconds

## BFS NS instance: NS_B2_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -145.9067841, 292.8096619, -150.8104706, 299.3354492, -445.2422485, 443.6200256
1: -53.6023979, 113.5242996, -55.8370361, 120.7850342, -174.3874359, 169.3613129
2: -26.1331673, 117.1856461, -26.8651829, 122.3678818, -148.5010529, 144.0507812
3: -63.6583138, 133.5087128, -67.1717453, 139.5548706, -203.2131500, 200.6804352
4: -33.3043480, 116.7929077, -34.5897942, 120.6810532, -153.9853821, 151.3826599

Time for backsubstitution: 3.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A2_A1_B2_B1

### Relational analysis result of NS_B2_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.2098761, upper bound: 319.6242191
time: 0.64 seconds

## Relational analysis of NS_B2_A1_A2_A1_B2_B2

### Relational analysis result of NS_B2_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.2098761, upper bound: 319.7126212
time: 0.66 seconds

## BFS NS instance: NS_B2_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -129.9613342, 268.2101440, -144.5021515, 288.9411926, -418.9025269, 412.7122803
1: -48.9637794, 103.4860077, -53.8252144, 116.1332703, -165.0970459, 157.3112183
2: -23.3542976, 108.0833130, -25.7462177, 117.9775848, -141.3318787, 133.8295135
3: -57.5968552, 121.4740753, -64.6137085, 134.1132355, -191.7100830, 186.0877838
4: -29.7535763, 107.3545532, -33.1719093, 116.2824936, -146.0360718, 140.5264587

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A2_A2_B1_B1

### Relational analysis result of NS_B2_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.1737096, upper bound: 318.7455165
time: 0.68 seconds

## Relational analysis of NS_B2_A1_A2_A2_B1_B2

### Relational analysis result of NS_B2_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.1737097, upper bound: 318.7455165
time: 0.71 seconds

## BFS NS instance: NS_B2_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -129.9613342, 268.2101440, -148.6785736, 295.7914124, -425.7527161, 416.8887329
1: -48.9637794, 103.4860077, -55.1593781, 119.2441025, -168.2078857, 158.6453857
2: -23.3542976, 108.0833130, -26.4833279, 120.9312363, -144.2855377, 134.5666351
3: -57.5968552, 121.4740753, -66.3092117, 137.7049255, -195.3017883, 187.7832947
4: -29.7535763, 107.3545532, -34.1019554, 119.2238312, -148.9774017, 141.4564972

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_B2_A1_A2_A2_B2_B1

### Relational analysis result of NS_B2_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.1737097, upper bound: 318.9756084
time: 0.71 seconds

## Relational analysis of NS_B2_A1_A2_A2_B2_B2

### Relational analysis result of NS_B2_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -318.1737096, upper bound: 318.9756084
time: 0.71 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.48 seconds
NS_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -319.0441021, upper bound: 319.0441021
NS_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -319.0441021, upper bound: 319.0441021
NS_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -319.0441021, upper bound: 319.0441021
NS_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -319.0441021, upper bound: 319.0441021
NS_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.9785179, upper bound: 318.2963916
NS_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.9785179, upper bound: 318.4030565
NS_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.9785179, upper bound: 318.2963916
NS_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.9785179, upper bound: 318.4030563
NS_B1_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.9558761, upper bound: 317.8847282
NS_B1_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.9558761, upper bound: 318.1691993
NS_B1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.9558761, upper bound: 317.8847282
NS_B1_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.9558761, upper bound: 318.1691993
NS_B1_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.5549443, upper bound: 317.6893191
NS_B1_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.5549443, upper bound: 318.0653022
NS_B1_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.5549443, upper bound: 317.6893191
NS_B1_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.5549443, upper bound: 318.0653022
NS_B1_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -319.0258194, upper bound: 318.6092716
NS_B1_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -319.0258194, upper bound: 318.6092716
NS_B1_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -319.0258194, upper bound: 318.6092716
NS_B1_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -319.0258194, upper bound: 318.6092716
NS_B1_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.5909889, upper bound: 318.5909889
NS_B1_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.5909889, upper bound: 318.5909889
NS_B1_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.5909889, upper bound: 318.5909889
NS_B1_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.5909889, upper bound: 318.5909889
NS_B2_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.2109227, upper bound: 319.6408668
NS_B2_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.2109227, upper bound: 319.7296873
NS_B2_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.2109227, upper bound: 319.6408668
NS_B2_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.2109227, upper bound: 319.7296873
NS_B2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.1698742, upper bound: 318.6362369
NS_B2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.1698742, upper bound: 318.6362369
NS_B2_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.1698742, upper bound: 318.6362369
NS_B2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.1698742, upper bound: 318.6362369
NS_B2_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.2098761, upper bound: 319.6242191
NS_B2_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.2098761, upper bound: 319.7126212
NS_B2_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.2098761, upper bound: 319.6242191
NS_B2_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.2098761, upper bound: 319.7126212
NS_B2_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.1737096, upper bound: 318.7455165
NS_B2_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.1737097, upper bound: 318.7455165
NS_B2_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.1737097, upper bound: 318.9756084
NS_B2_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.48
Output dim: 0, lower bound: -318.1737096, upper bound: 318.9756084

## BFS NS instance: NS_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -146.0284119, 293.1794434, -146.0284119, 293.1794434, -439.2078247, 439.2078247
1: -53.6530647, 113.5495758, -53.6530647, 113.5495758, -167.2026367, 167.2026367
2: -26.1678429, 117.2793808, -26.1678429, 117.2793808, -143.4472198, 143.4472198
3: -63.6867294, 133.6244202, -63.6867294, 133.6244202, -197.3111572, 197.3111572
4: -33.3424187, 116.9046783, -33.3424187, 116.9046783, -150.2471008, 150.2471008

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

## BFS NS instance: NS_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -130.1889343, 268.6842041, -146.0284119, 293.1794434, -423.3683777, 414.7126160
1: -49.0288544, 103.4790955, -53.6530647, 113.5495758, -162.5784302, 157.1321411
2: -23.4022141, 108.1495972, -26.1678429, 117.2793808, -140.6815796, 134.3174438
3: -57.6666183, 121.6073227, -63.6867294, 133.6244202, -191.2910461, 185.2940369
4: -29.8077126, 107.4550095, -33.3424187, 116.9046783, -146.7123871, 140.7974091

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

## BFS NS instance: NS_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -146.0284119, 293.1794434, -130.1889343, 268.6842041, -414.7126160, 423.3683777
1: -53.6530647, 113.5495758, -49.0288544, 103.4790955, -157.1321564, 162.5784302
2: -26.1678429, 117.2793808, -23.4022141, 108.1495972, -134.3174438, 140.6815796
3: -63.6867294, 133.6244202, -57.6666183, 121.6073227, -185.2940521, 191.2910461
4: -33.3424187, 116.9046783, -29.8077126, 107.4550095, -140.7974091, 146.7123871

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

## BFS NS instance: NS_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -130.1889343, 268.6842041, -130.1889343, 268.6842041, -398.8731384, 398.8731384
1: -49.0288544, 103.4790955, -49.0288544, 103.4790955, -152.5079498, 152.5079498
2: -23.4022141, 108.1495972, -23.4022141, 108.1495972, -131.5518036, 131.5518036
3: -57.6666183, 121.6073227, -57.6666183, 121.6073227, -179.2739410, 179.2739410
4: -29.8077126, 107.4550095, -29.8077126, 107.4550095, -137.2627106, 137.2627106

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

## BFS NS instance: NS_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -148.0646210, 294.6705627, -146.0284119, 293.1794434, -441.2440796, 440.6989746
1: -54.9366188, 118.6856918, -53.6530647, 113.5495758, -168.4861908, 172.3387299
2: -26.3615475, 120.3615646, -26.1678429, 117.2793808, -143.6409149, 146.5294037
3: -66.0272141, 137.1218567, -63.6867294, 133.6244202, -199.6516266, 200.8085938
4: -33.9626846, 118.6733246, -33.3424187, 116.9046783, -150.8673706, 152.0157471

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 27

## BFS NS instance: NS_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -135.9183350, 276.5257874, -146.0284119, 293.1794434, -429.0977783, 422.5541687
1: -51.5104942, 110.7328186, -53.6530647, 113.5495758, -165.0600739, 164.3858643
2: -24.2533665, 113.9976044, -26.1678429, 117.2793808, -141.5327454, 140.1654510
3: -61.3505402, 128.0840759, -63.6867294, 133.6244202, -194.9749603, 191.7708130
4: -31.2086792, 112.0452728, -33.3424187, 116.9046783, -148.1133575, 145.3876801

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 27

## BFS NS instance: NS_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -148.0646210, 294.6705627, -130.1889343, 268.6842041, -416.7488403, 424.8594971
1: -54.9366188, 118.6856918, -49.0288544, 103.4790955, -158.4157104, 167.7145386
2: -26.3615475, 120.3615646, -23.4022141, 108.1495972, -134.5111389, 143.7637634
3: -66.0272141, 137.1218567, -57.6666183, 121.6073227, -187.6344910, 194.7884827
4: -33.9626846, 118.6733246, -29.8077126, 107.4550095, -141.4176941, 148.4810333

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## BFS NS instance: NS_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -135.9183350, 276.5257874, -130.1889343, 268.6842041, -404.6025391, 406.7147217
1: -51.5104942, 110.7328186, -49.0288544, 103.4790955, -154.9895782, 159.7616577
2: -24.2533665, 113.9976044, -23.4022141, 108.1495972, -132.4029694, 137.3997955
3: -61.3505402, 128.0840759, -57.6666183, 121.6073227, -182.9578552, 185.7507019
4: -31.2086792, 112.0452728, -29.8077126, 107.4550095, -138.6636658, 141.8529816

Time for backsubstitution: 3.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

## BFS NS instance: NS_B1_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -146.0284119, 293.1794434, -151.1171875, 300.9364929, -446.9648438, 444.2966003
1: -53.6530647, 113.5495758, -55.0382729, 116.7512741, -170.4043121, 168.5878448
2: -26.1678429, 117.2793808, -27.0332279, 120.1250458, -146.2928925, 144.3126068
3: -63.6867294, 133.6244202, -65.6367645, 137.1298981, -200.8166199, 199.2611694
4: -33.3424187, 116.9046783, -34.4027023, 119.8971176, -153.2395325, 151.3073730

Time for backsubstitution: 3.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

## BFS NS instance: NS_B1_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -130.1889343, 268.6842041, -151.1171875, 300.9364929, -431.1254272, 419.8013916
1: -49.0288544, 103.4790955, -55.0382729, 116.7512741, -165.7801208, 158.5173645
2: -23.4022141, 108.1495972, -27.0332279, 120.1250458, -143.5272217, 135.1828308
3: -57.6666183, 121.6073227, -65.6367645, 137.1298981, -194.7965088, 187.2440338
4: -29.8077126, 107.4550095, -34.4027023, 119.8971176, -149.7048340, 141.8577118

Time for backsubstitution: 3.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

## BFS NS instance: NS_B1_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -148.0646210, 294.6705627, -151.1171875, 300.9364929, -449.0010986, 445.7877197
1: -54.9366188, 118.6856918, -55.0382729, 116.7512741, -171.6878967, 173.7239685
2: -26.3615475, 120.3615646, -27.0332279, 120.1250458, -146.4865723, 147.3947906
3: -66.0272141, 137.1218567, -65.6367645, 137.1298981, -203.1571045, 202.7586060
4: -33.9626846, 118.6733246, -34.4027023, 119.8971176, -153.8598022, 153.0760193

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 27

## BFS NS instance: NS_B1_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -135.9183350, 276.5257874, -151.1171875, 300.9364929, -436.8547974, 427.6429443
1: -51.5104942, 110.7328186, -55.0382729, 116.7512741, -168.2617493, 165.7710876
2: -24.2533665, 113.9976044, -27.0332279, 120.1250458, -144.3784027, 141.0308380
3: -61.3505402, 128.0840759, -65.6367645, 137.1298981, -198.4804382, 193.7208099
4: -31.2086792, 112.0452728, -34.4027023, 119.8971176, -151.1058044, 146.4479675

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 27

## BFS NS instance: NS_B1_A1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -146.0284119, 293.1794434, -134.7574463, 275.9481506, -421.9765625, 427.9368591
1: -53.6530647, 113.5495758, -50.3087044, 105.9661560, -159.6192169, 163.8582764
2: -26.1678429, 117.2793808, -24.1924667, 110.6458969, -136.8137360, 141.4718323
3: -63.6867294, 133.6244202, -59.3704910, 124.7822495, -188.4689789, 192.9949036
4: -33.3424187, 116.9046783, -30.7387390, 110.2618256, -143.6042480, 147.6434174

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

## BFS NS instance: NS_B1_A1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -130.1889343, 268.6842041, -134.7574463, 275.9481506, -406.1370850, 403.4416504
1: -49.0288544, 103.4790955, -50.3087044, 105.9661560, -154.9950104, 153.7877960
2: -23.4022141, 108.1495972, -24.1924667, 110.6458969, -134.0481110, 132.3420563
3: -57.6666183, 121.6073227, -59.3704910, 124.7822495, -182.4488678, 180.9778137
4: -29.8077126, 107.4550095, -30.7387390, 110.2618256, -140.0695343, 138.1937256

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

## BFS NS instance: NS_B1_A1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -148.0646210, 294.6705627, -134.7574463, 275.9481506, -424.0127563, 429.4279785
1: -54.9366188, 118.6856918, -50.3087044, 105.9661560, -160.9027710, 168.9944000
2: -26.3615475, 120.3615646, -24.1924667, 110.6458969, -137.0074463, 144.5540161
3: -66.0272141, 137.1218567, -59.3704910, 124.7822495, -190.8094482, 196.4923401
4: -33.9626846, 118.6733246, -30.7387390, 110.2618256, -144.2245178, 149.4120636

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

## BFS NS instance: NS_B1_A1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -135.9183350, 276.5257874, -134.7574463, 275.9481506, -411.8664856, 411.2832031
1: -51.5104942, 110.7328186, -50.3087044, 105.9661560, -157.4766541, 161.0415039
2: -24.2533665, 113.9976044, -24.1924667, 110.6458969, -134.8992615, 138.1900482
3: -61.3505402, 128.0840759, -59.3704910, 124.7822495, -186.1327820, 187.4545593
4: -31.2086792, 112.0452728, -30.7387390, 110.2618256, -141.4705048, 142.7840118

Time for backsubstitution: 3.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 27

## BFS NS instance: NS_B1_A2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -151.1171875, 300.9364929, -146.0284119, 293.1794434, -444.2966003, 446.9648743
1: -55.0382729, 116.7512741, -53.6530647, 113.5495758, -168.5878448, 170.4043121
2: -27.0332279, 120.1250458, -26.1678429, 117.2793808, -144.3126068, 146.2928772
3: -65.6367645, 137.1298981, -63.6867294, 133.6244202, -199.2611542, 200.8166199
4: -34.4027023, 119.8971176, -33.3424187, 116.9046783, -151.3073730, 153.2395325

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

## BFS NS instance: NS_B1_A2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -151.1171875, 300.9364929, -130.1889343, 268.6842041, -419.8013916, 431.1254272
1: -55.0382729, 116.7512741, -49.0288544, 103.4790955, -158.5173645, 165.7801208
2: -27.0332279, 120.1250458, -23.4022141, 108.1495972, -135.1828308, 143.5272217
3: -65.6367645, 137.1298981, -57.6666183, 121.6073227, -187.2440186, 194.7965088
4: -34.4027023, 119.8971176, -29.8077126, 107.4550095, -141.8577118, 149.7048187

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

## BFS NS instance: NS_B1_A2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -134.7574463, 275.9481506, -146.0284119, 293.1794434, -427.9368591, 421.9765320
1: -50.3087044, 105.9661560, -53.6530647, 113.5495758, -163.8582764, 159.6192169
2: -24.1924667, 110.6458969, -26.1678429, 117.2793808, -141.4718323, 136.8137360
3: -59.3704910, 124.7822495, -63.6867294, 133.6244202, -192.9949036, 188.4689789
4: -30.7387390, 110.2618256, -33.3424187, 116.9046783, -147.6434174, 143.6042480

Time for backsubstitution: 2.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

## BFS NS instance: NS_B1_A2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -134.7574463, 275.9481506, -130.1889343, 268.6842041, -403.4416504, 406.1370850
1: -50.3087044, 105.9661560, -49.0288544, 103.4790955, -153.7877960, 154.9950104
2: -24.1924667, 110.6458969, -23.4022141, 108.1495972, -132.3420563, 134.0481110
3: -59.3704910, 124.7822495, -57.6666183, 121.6073227, -180.9778137, 182.4488678
4: -30.7387390, 110.2618256, -29.8077126, 107.4550095, -138.1937256, 140.0695343

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

## BFS NS instance: NS_B1_A2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -151.1171875, 300.9364929, -150.8536224, 300.4692383, -451.5864258, 451.7901001
1: -55.0382729, 116.7512741, -54.9500847, 116.5752869, -171.6135559, 171.7013550
2: -27.0332279, 120.1250458, -26.9824333, 119.9479828, -146.9812164, 147.1074829
3: -65.6367645, 137.1298981, -65.5340042, 136.9148254, -202.5515289, 202.6639099
4: -34.4027023, 119.8971176, -34.3438072, 119.7164993, -154.1192017, 154.2409210

Time for backsubstitution: 2.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

## BFS NS instance: NS_B1_A2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -151.1171875, 300.9364929, -134.5044556, 275.5016479, -426.6188354, 435.4409485
1: -55.0382729, 116.7512741, -50.2241440, 105.7980423, -160.8363190, 166.9754181
2: -27.0332279, 120.1250458, -24.1433125, 110.4786224, -137.5118561, 144.2683411
3: -65.6367645, 137.1298981, -59.2745934, 124.5759811, -190.2127380, 196.4044952
4: -34.4027023, 119.8971176, -30.6818123, 110.0898514, -144.4925537, 150.5789337

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

## BFS NS instance: NS_B1_A2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -134.7574463, 275.9481506, -150.8536224, 300.4692383, -435.2266846, 426.8017578
1: -50.3087044, 105.9661560, -54.9500847, 116.5752869, -166.8839874, 160.9162445
2: -24.1924667, 110.6458969, -26.9824333, 119.9479828, -144.1404419, 137.6283264
3: -59.3704910, 124.7822495, -65.5340042, 136.9148254, -196.2852936, 190.3162537
4: -30.7387390, 110.2618256, -34.3438072, 119.7164993, -150.4552307, 144.6056366

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

## BFS NS instance: NS_B1_A2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -134.7574463, 275.9481506, -134.5044556, 275.5016479, -410.2590942, 410.4526062
1: -50.3087044, 105.9661560, -50.2241440, 105.7980423, -156.1067505, 156.1903076
2: -24.1924667, 110.6458969, -24.1433125, 110.4786224, -134.6710663, 134.7892151
3: -59.3704910, 124.7822495, -59.2745934, 124.5759811, -183.9464722, 184.0568390
4: -30.7387390, 110.2618256, -30.6818123, 110.0898514, -140.8285828, 140.9436340

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

## BFS NS instance: NS_B2_A1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -140.9439850, 284.5309753, -145.8747101, 291.1589355, -432.1029053, 430.4057007
1: -52.0243225, 109.8980255, -54.2546692, 117.1284180, -169.1526947, 164.1526794
2: -25.2635994, 113.7951508, -25.9917545, 118.8756104, -144.1391907, 139.7868958
3: -61.6253319, 129.3824921, -65.1639023, 135.3112793, -196.9365692, 194.5463867
4: -32.2032166, 113.3641968, -33.4882660, 117.1914673, -149.3946838, 146.8524628

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_B2_A1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -140.9439850, 284.5309753, -133.8505249, 273.3176880, -414.2616577, 418.3815002
1: -52.0243225, 109.8980255, -50.8837509, 109.3193588, -161.3436127, 160.7817688
2: -25.2635994, 113.7951508, -23.9138336, 112.6632156, -137.9267883, 137.7089844
3: -61.6253319, 129.3824921, -60.5607796, 126.4849930, -188.1103210, 189.9432678
4: -32.2032166, 113.3641968, -30.7754459, 110.7111664, -142.9143677, 144.1396332

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_B2_A1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -140.9439850, 284.5309753, -150.1038055, 298.1235657, -439.0675659, 434.6347656
1: -52.0243225, 109.8980255, -55.6091805, 120.2820587, -172.3063812, 165.5072021
2: -25.2635994, 113.7951508, -26.7391796, 121.8825684, -147.1461639, 140.5343018
3: -61.6253319, 129.3824921, -66.8841019, 138.9542236, -200.5795593, 196.2666016
4: -32.2032166, 113.3641968, -34.4308662, 120.1877518, -152.3909302, 147.7950287

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_B2_A1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -140.9439850, 284.5309753, -137.5089264, 279.2254028, -420.1693726, 422.0399170
1: -52.0243225, 109.8980255, -52.0336304, 112.0259628, -164.0502625, 161.9316406
2: -25.2635994, 113.7951508, -24.5523949, 115.1943817, -140.4579773, 138.3475342
3: -61.6253319, 129.3824921, -62.0294876, 129.5164642, -191.1417999, 191.4119873
4: -32.2032166, 113.3641968, -31.5776806, 113.2390060, -145.4422302, 144.9418640

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_B2_A1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -125.5645752, 261.0166626, -145.8747101, 291.1589355, -416.7234192, 406.8913574
1: -47.5888519, 100.3641891, -54.2546692, 117.1284180, -164.7172546, 154.6188354
2: -22.5948162, 105.0696259, -25.9917545, 118.8756104, -141.4704285, 131.0613708
3: -55.8606834, 117.8552094, -65.1639023, 135.3112793, -191.1719208, 183.0191040
4: -28.8149567, 104.3087234, -33.4882660, 117.1914673, -146.0064240, 137.7969971

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

## BFS NS instance: NS_B2_A1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -125.5645752, 261.0166626, -133.8505249, 273.3176880, -398.8822632, 394.8671875
1: -47.5888519, 100.3641891, -50.8837509, 109.3193588, -156.9081879, 151.2479401
2: -22.5948162, 105.0696259, -23.9138336, 112.6632156, -135.2580261, 128.9834595
3: -55.8606834, 117.8552094, -60.5607796, 126.4849930, -182.3456726, 178.4159851
4: -28.8149567, 104.3087234, -30.7754459, 110.7111664, -139.5261230, 135.0841675

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_B2_A1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -125.5645752, 261.0166626, -150.1038055, 298.1235657, -423.6881104, 411.1204834
1: -47.5888519, 100.3641891, -55.6091805, 120.2820587, -167.8709106, 155.9733582
2: -22.5948162, 105.0696259, -26.7391796, 121.8825684, -144.4773865, 131.8087921
3: -55.8606834, 117.8552094, -66.8841019, 138.9542236, -194.8149109, 184.7393188
4: -28.8149567, 104.3087234, -34.4308662, 120.1877518, -149.0026855, 138.7395935

Time for backsubstitution: 3.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

## BFS NS instance: NS_B2_A1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -125.5645752, 261.0166626, -137.5089264, 279.2254028, -404.7899475, 398.5255737
1: -47.5888519, 100.3641891, -52.0336304, 112.0259628, -159.6148071, 152.3978119
2: -22.5948162, 105.0696259, -24.5523949, 115.1943817, -137.7891998, 129.6220093
3: -55.8606834, 117.8552094, -62.0294876, 129.5164642, -185.3771515, 179.8847046
4: -28.8149567, 104.3087234, -31.5776806, 113.2390060, -142.0539551, 135.8863983

Time for backsubstitution: 3.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_B2_A1_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -145.9067841, 292.8096619, -145.8747101, 291.1589355, -437.0657349, 438.6843567
1: -53.6023979, 113.5242996, -54.2546692, 117.1284180, -170.7308044, 167.7789459
2: -26.1331673, 117.1856461, -25.9917545, 118.8756104, -145.0087738, 143.1773682
3: -63.6583138, 133.5087128, -65.1639023, 135.3112793, -198.9695587, 198.6726074
4: -33.3043480, 116.7929077, -33.4882660, 117.1914673, -150.4958191, 150.2811737

Time for backsubstitution: 2.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_B2_A1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -145.9067841, 292.8096619, -133.8505249, 273.3176880, -419.2244873, 426.6601868
1: -53.6023979, 113.5242996, -50.8837509, 109.3193588, -162.9217224, 164.4080353
2: -26.1331673, 117.1856461, -23.9138336, 112.6632156, -138.7963867, 141.0994568
3: -63.6583138, 133.5087128, -60.5607796, 126.4849930, -190.1433105, 194.0694885
4: -33.3043480, 116.7929077, -30.7754459, 110.7111664, -144.0154877, 147.5683289

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_B2_A1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -145.9067841, 292.8096619, -150.1038055, 298.1235657, -444.0303345, 442.9134521
1: -53.6023979, 113.5242996, -55.6091805, 120.2820587, -173.8844604, 169.1334686
2: -26.1331673, 117.1856461, -26.7391796, 121.8825684, -148.0157318, 143.9248047
3: -63.6583138, 133.5087128, -66.8841019, 138.9542236, -202.6125336, 200.3928223
4: -33.3043480, 116.7929077, -34.4308662, 120.1877518, -153.4920654, 151.2237396

Time for backsubstitution: 2.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_B2_A1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -145.9067841, 292.8096619, -137.5089264, 279.2254028, -425.1322021, 430.3185730
1: -53.6023979, 113.5242996, -52.0336304, 112.0259628, -165.6283569, 165.5578918
2: -26.1331673, 117.1856461, -24.5523949, 115.1943817, -141.3275452, 141.7380219
3: -63.6583138, 133.5087128, -62.0294876, 129.5164642, -193.1747742, 195.5382080
4: -33.3043480, 116.7929077, -31.5776806, 113.2390060, -146.5433502, 148.3705750

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_B2_A1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -129.9613342, 268.2101440, -145.8747101, 291.1589355, -421.1202393, 414.0848389
1: -48.9637794, 103.4860077, -54.2546692, 117.1284180, -166.0921936, 157.7406464
2: -23.3542976, 108.0833130, -25.9917545, 118.8756104, -142.2299042, 134.0750427
3: -57.5968552, 121.4740753, -65.1639023, 135.3112793, -192.9081268, 186.6379700
4: -29.7535763, 107.3545532, -33.4882660, 117.1914673, -146.9450378, 140.8428040

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

## BFS NS instance: NS_B2_A1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -129.9613342, 268.2101440, -133.8505249, 273.3176880, -403.2790222, 402.0606689
1: -48.9637794, 103.4860077, -50.8837509, 109.3193588, -158.2831116, 154.3697357
2: -23.3542976, 108.0833130, -23.9138336, 112.6632156, -136.0175171, 131.9971313
3: -57.5968552, 121.4740753, -60.5607796, 126.4849930, -184.0818481, 182.0348511
4: -29.7535763, 107.3545532, -30.7754459, 110.7111664, -140.4647369, 138.1299896

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_B2_A1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -129.9613342, 268.2101440, -150.1038055, 298.1235657, -428.0848999, 418.3139648
1: -48.9637794, 103.4860077, -55.6091805, 120.2820587, -169.2458344, 159.0951691
2: -23.3542976, 108.0833130, -26.7391796, 121.8825684, -145.2368622, 134.8224640
3: -57.5968552, 121.4740753, -66.8841019, 138.9542236, -196.5510864, 188.3581848
4: -29.7535763, 107.3545532, -34.4308662, 120.1877518, -149.9412994, 141.7854004

Time for backsubstitution: 3.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 29

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

## BFS NS instance: NS_B2_A1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -129.9613342, 268.2101440, -137.5089264, 279.2254028, -409.1867371, 405.7190552
1: -48.9637794, 103.4860077, -52.0336304, 112.0259628, -160.9897461, 155.5196075
2: -23.3542976, 108.0833130, -24.5523949, 115.1943817, -138.5486755, 132.6356812
3: -57.5968552, 121.4740753, -62.0294876, 129.5164642, -187.1133118, 183.5035706
4: -29.7535763, 107.3545532, -31.5776806, 113.2390060, -142.9925842, 138.9322052

Time for backsubstitution: 2.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 27

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.91 + 330.95 = 334.85 seconds
