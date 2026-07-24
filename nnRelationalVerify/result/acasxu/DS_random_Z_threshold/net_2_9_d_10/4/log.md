## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 14.783633487000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007)
1: (-10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563)
2: (-6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616)
3: (-7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103)
4: (-5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.01 + 1.35 = 2.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -14.8206852, upper bound: 14.8206852

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.36 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.72 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7607383, upper bound: 14.7607383
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7607383, upper bound: 14.7607383
time: 0.32 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.35 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.68 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 1.68
Output dim: 4, lower bound: -14.7607383, upper bound: 14.7607383
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 1.68
Output dim: 4, lower bound: -14.7607383, upper bound: 14.7607383
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.68
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.68
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 10

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.35 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.66 seconds
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.66
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.66
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.66
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.66
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7856875, upper bound: 14.7856875
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7856875, upper bound: 14.7856875
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8148004, upper bound: 14.8148004
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8148004, upper bound: 14.8148004
time: 0.35 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.73 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.73
Output dim: 4, lower bound: -14.7856875, upper bound: 14.7856875
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.73
Output dim: 4, lower bound: -14.7856875, upper bound: 14.7856875
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.73
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.73
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.73
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.73
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.73
Output dim: 4, lower bound: -14.8148004, upper bound: 14.8148004
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.73
Output dim: 4, lower bound: -14.8148004, upper bound: 14.8148004

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7819945, upper bound: 14.7819945
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7819945, upper bound: 14.7819945
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7819945, upper bound: 14.7819945
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7819945, upper bound: 14.7819945
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7268101, upper bound: 14.7268101
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7268101, upper bound: 14.7268101
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8148004, upper bound: 14.8148004
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8148004, upper bound: 14.8148004
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8126064, upper bound: 14.8126064
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8126064, upper bound: 14.8126064
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8088006, upper bound: 14.8088006
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8088006, upper bound: 14.8088006
time: 0.34 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.72 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.72
Output dim: 4, lower bound: -14.7819945, upper bound: 14.7819945
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.72
Output dim: 4, lower bound: -14.7819945, upper bound: 14.7819945
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.72
Output dim: 4, lower bound: -14.7819945, upper bound: 14.7819945
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.72
Output dim: 4, lower bound: -14.7819945, upper bound: 14.7819945
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.72
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.72
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.72
Output dim: 4, lower bound: -14.7268101, upper bound: 14.7268101
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.72
Output dim: 4, lower bound: -14.7268101, upper bound: 14.7268101
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.72
Output dim: 4, lower bound: -14.8148004, upper bound: 14.8148004
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.72
Output dim: 4, lower bound: -14.8148004, upper bound: 14.8148004
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.72
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.72
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.72
Output dim: 4, lower bound: -14.8126064, upper bound: 14.8126064
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.72
Output dim: 4, lower bound: -14.8126064, upper bound: 14.8126064
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 1.72
Output dim: 4, lower bound: -14.8088006, upper bound: 14.8088006
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 1.72
Output dim: 4, lower bound: -14.8088006, upper bound: 14.8088006

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7771504, upper bound: 14.7771504
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7771504, upper bound: 14.7771504
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7771504, upper bound: 14.7771504
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7771504, upper bound: 14.7771504
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7488376, upper bound: 14.7488376
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7488376, upper bound: 14.7488376
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8148004, upper bound: 14.8148004
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8148004, upper bound: 14.8148004
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8118414, upper bound: 14.8118414
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8118414, upper bound: 14.8118414
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8066240, upper bound: 14.8066240
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8066240, upper bound: 14.8066240
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8066240, upper bound: 14.8066240
time: 0.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8066240, upper bound: 14.8066240
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7459843, upper bound: 14.7459843
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7459843, upper bound: 14.7459843
time: 0.34 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 1.71 seconds
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.7771504, upper bound: 14.7771504
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.7771504, upper bound: 14.7771504
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.7771504, upper bound: 14.7771504
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.7771504, upper bound: 14.7771504
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.7488376, upper bound: 14.7488376
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.7488376, upper bound: 14.7488376
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.8148004, upper bound: 14.8148004
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.8148004, upper bound: 14.8148004
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.8118414, upper bound: 14.8118414
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.8118414, upper bound: 14.8118414
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.8066240, upper bound: 14.8066240
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.8066240, upper bound: 14.8066240
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.8066240, upper bound: 14.8066240
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.8066240, upper bound: 14.8066240
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.7459843, upper bound: 14.7459843
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 1.71
Output dim: 4, lower bound: -14.7459843, upper bound: 14.7459843

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7488376, upper bound: 14.7488376
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7488376, upper bound: 14.7488376
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8088006, upper bound: 14.8088006
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8088006, upper bound: 14.8088006
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8098580, upper bound: 14.8098580
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8098580, upper bound: 14.8098580
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8146805, upper bound: 14.8146805
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8146805, upper bound: 14.8146805
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8191061, upper bound: 14.8191061
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8191061, upper bound: 14.8191061
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8118414, upper bound: 14.8118414
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8118414, upper bound: 14.8118414
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8111654, upper bound: 14.8111654
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8111654, upper bound: 14.8111654
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8046343, upper bound: 14.8046343
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8046343, upper bound: 14.8046343
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8057817, upper bound: 14.8057817
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8057817, upper bound: 14.8057817
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8046343, upper bound: 14.8046343
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8046343, upper bound: 14.8046343
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8057817, upper bound: 14.8057817
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8057817, upper bound: 14.8057817
time: 0.36 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 1.76 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.7488376, upper bound: 14.7488376
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.7488376, upper bound: 14.7488376
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8088006, upper bound: 14.8088006
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8088006, upper bound: 14.8088006
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8098580, upper bound: 14.8098580
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8098580, upper bound: 14.8098580
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8199747, upper bound: 14.8199747
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8146805, upper bound: 14.8146805
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8146805, upper bound: 14.8146805
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8191061, upper bound: 14.8191061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8191061, upper bound: 14.8191061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8118414, upper bound: 14.8118414
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8118414, upper bound: 14.8118414
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8111654, upper bound: 14.8111654
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8111654, upper bound: 14.8111654
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8046343, upper bound: 14.8046343
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8046343, upper bound: 14.8046343
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8057817, upper bound: 14.8057817
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8057817, upper bound: 14.8057817
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8046343, upper bound: 14.8046343
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8046343, upper bound: 14.8046343
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8057817, upper bound: 14.8057817
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 1.76
Output dim: 4, lower bound: -14.8057817, upper bound: 14.8057817

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8080209, upper bound: 14.8080209
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8080209, upper bound: 14.8080209
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8084811, upper bound: 14.8084811
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8084811, upper bound: 14.8084811
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8146805, upper bound: 14.8146805
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8146805, upper bound: 14.8146805
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8191061, upper bound: 14.8191061
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8191061, upper bound: 14.8191061
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8187799, upper bound: 14.8187799
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8187799, upper bound: 14.8187799
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8138863, upper bound: 14.8138863
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8138863, upper bound: 14.8138863
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8057817, upper bound: 14.8057817
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8057817, upper bound: 14.8057817
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8057817, upper bound: 14.8057817
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8057817, upper bound: 14.8057817
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8073950, upper bound: 14.8073950
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8073950, upper bound: 14.8073950
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8111654, upper bound: 14.8111654
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8111654, upper bound: 14.8111654
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8046343, upper bound: 14.8046343
time: 0.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8046343, upper bound: 14.8046343
time: 0.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8002312, upper bound: 14.8002312
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8002312, upper bound: 14.8002312
time: 0.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
time: 0.37 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 2.35 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8080209, upper bound: 14.8080209
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8080209, upper bound: 14.8080209
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8084811, upper bound: 14.8084811
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8084811, upper bound: 14.8084811
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8146805, upper bound: 14.8146805
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8146805, upper bound: 14.8146805
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8191061, upper bound: 14.8191061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8191061, upper bound: 14.8191061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8187799, upper bound: 14.8187799
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8187799, upper bound: 14.8187799
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8138863, upper bound: 14.8138863
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8138863, upper bound: 14.8138863
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8057817, upper bound: 14.8057817
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8057817, upper bound: 14.8057817
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8057817, upper bound: 14.8057817
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8057817, upper bound: 14.8057817
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8073950, upper bound: 14.8073950
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8073950, upper bound: 14.8073950
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8111654, upper bound: 14.8111654
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8111654, upper bound: 14.8111654
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8046343, upper bound: 14.8046343
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8046343, upper bound: 14.8046343
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8002312, upper bound: 14.8002312
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8002312, upper bound: 14.8002312
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 2.35
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8016532, upper bound: 14.8016532
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8016532, upper bound: 14.8016532
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8016532, upper bound: 14.8016532
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8016532, upper bound: 14.8016532
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8083482, upper bound: 14.8083482
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8083482, upper bound: 14.8083482
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8084811, upper bound: 14.8084811
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8084811, upper bound: 14.8084811
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8138863, upper bound: 14.8138863
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8138863, upper bound: 14.8138863
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8145415, upper bound: 14.8145415
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8145415, upper bound: 14.8145415
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8138863, upper bound: 14.8138863
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8138863, upper bound: 14.8138863
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8187799, upper bound: 14.8187799
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8187799, upper bound: 14.8187799
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8138863, upper bound: 14.8138863
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8138863, upper bound: 14.8138863
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8073950, upper bound: 14.8073950
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8073950, upper bound: 14.8073950
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8073950, upper bound: 14.8073950
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8073950, upper bound: 14.8073950
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8002312, upper bound: 14.8002312
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8002312, upper bound: 14.8002312
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8002312, upper bound: 14.8002312
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8002312, upper bound: 14.8002312
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8002312, upper bound: 14.8002312
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8002312, upper bound: 14.8002312
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.40 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 2.02 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8016532, upper bound: 14.8016532
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8016532, upper bound: 14.8016532
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8016532, upper bound: 14.8016532
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8016532, upper bound: 14.8016532
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8083482, upper bound: 14.8083482
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8083482, upper bound: 14.8083482
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8084811, upper bound: 14.8084811
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8084811, upper bound: 14.8084811
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8138863, upper bound: 14.8138863
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8138863, upper bound: 14.8138863
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8145415, upper bound: 14.8145415
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8145415, upper bound: 14.8145415
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8138863, upper bound: 14.8138863
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8138863, upper bound: 14.8138863
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8187799, upper bound: 14.8187799
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8187799, upper bound: 14.8187799
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8138863, upper bound: 14.8138863
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8138863, upper bound: 14.8138863
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8073950, upper bound: 14.8073950
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8073950, upper bound: 14.8073950
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8073950, upper bound: 14.8073950
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8073950, upper bound: 14.8073950
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8002312, upper bound: 14.8002312
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8002312, upper bound: 14.8002312
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8002312, upper bound: 14.8002312
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8002312, upper bound: 14.8002312
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8002312, upper bound: 14.8002312
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8002312, upper bound: 14.8002312
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8035491, upper bound: 14.8035491
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8051058, upper bound: 14.8051058
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 2.02
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8083482, upper bound: 14.8083482
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8083482, upper bound: 14.8083482
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8083482, upper bound: 14.8083482
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8083482, upper bound: 14.8083482
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8076727, upper bound: 14.8076727
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8076727, upper bound: 14.8076727
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8083482, upper bound: 14.8083482
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8083482, upper bound: 14.8083482
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
time: 0.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7994452, upper bound: 14.7994452
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7994452, upper bound: 14.7994452
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7994452, upper bound: 14.7994452
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7994452, upper bound: 14.7994452
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7994452, upper bound: 14.7994452
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7994452, upper bound: 14.7994452
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7994452, upper bound: 14.7994452
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7994452, upper bound: 14.7994452
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 14

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
time: 0.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 14

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 40

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 23

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Candidate
type: DSZ, layer: 1, pos: 38

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
time: 0.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
time: 0.38 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 2.16 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8083482, upper bound: 14.8083482
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8083482, upper bound: 14.8083482
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8083482, upper bound: 14.8083482
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8083482, upper bound: 14.8083482
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8076727, upper bound: 14.8076727
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8076727, upper bound: 14.8076727
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8083482, upper bound: 14.8083482
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8083482, upper bound: 14.8083482
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8137196, upper bound: 14.8137196
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8048889, upper bound: 14.8048889
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7994452, upper bound: 14.7994452
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7994452, upper bound: 14.7994452
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7994452, upper bound: 14.7994452
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7994452, upper bound: 14.7994452
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7994452, upper bound: 14.7994452
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7994452, upper bound: 14.7994452
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7994452, upper bound: 14.7994452
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7994452, upper bound: 14.7994452
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7991061, upper bound: 14.7991061
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8008672, upper bound: 14.8008672
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.7983201, upper bound: 14.7983201
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 2.16
Output dim: 4, lower bound: -14.8027695, upper bound: 14.8027695

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 1.26 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 2.36 + 418.13 = 420.49 seconds
