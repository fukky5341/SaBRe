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
execution time: IAR + RelationalAnalysis = 2.88 + 1.50 = 4.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -14.8206852, upper bound: 14.8206852

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8127883, upper bound: 14.8127883
time: 0.45 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8127883, upper bound: 14.8127883
time: 0.52 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.21 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 4, lower bound: -14.8127883, upper bound: 14.8127883
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.21
Output dim: 4, lower bound: -14.8127883, upper bound: 14.8127883

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8124531, upper bound: 14.8124531
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8124531, upper bound: 14.8124531
time: 0.48 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 28

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8124531, upper bound: 14.8124531
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8124531, upper bound: 14.8124531
time: 0.50 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.92 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.92
Output dim: 4, lower bound: -14.8124531, upper bound: 14.8124531
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.92
Output dim: 4, lower bound: -14.8124531, upper bound: 14.8124531
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.92
Output dim: 4, lower bound: -14.8124531, upper bound: 14.8124531
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.92
Output dim: 4, lower bound: -14.8124531, upper bound: 14.8124531

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.50 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.92 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.92
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 2

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
time: 0.50 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.95 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 4, lower bound: -14.8061286, upper bound: 14.8061286

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
time: 0.50 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.03 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.03
Output dim: 4, lower bound: -14.8058786, upper bound: 14.8058786

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 3.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -149.4919434, 98.3400955, -149.4919434, 98.3400955, -247.8320007, 247.8320007
1: -10.2893686, 7.6158886, -10.2893686, 7.6158886, -17.9052544, 17.9052563
2: -6.0750151, 12.7892494, -6.0750151, 12.7892494, -18.8642616, 18.8642616
3: -7.9261875, 20.4676228, -7.9261875, 20.4676228, -28.3938103, 28.3938103
4: -5.2263708, 13.2979355, -5.2263708, 13.2979355, -18.5243073, 18.5243073

Time for backsubstitution: 2.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
time: 0.44 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.66 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.66
Output dim: 4, lower bound: -14.7601576, upper bound: 14.7601576

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.39 + 268.81 = 273.19 seconds
