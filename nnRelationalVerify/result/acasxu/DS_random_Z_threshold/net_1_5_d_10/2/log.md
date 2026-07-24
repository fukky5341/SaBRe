## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.088187946


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102)
1: (-0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898)
2: (-0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237)
3: (-0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035)
4: (-0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.10 + 0.87 = 1.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899608
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899296
time: 0.24 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 0.54 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 0.54
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899608
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 0.54
Output dim: 0, lower bound: -0.0899296, upper bound: 0.0899296

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899482
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899423
time: 0.26 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 48
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 48

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899088
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899066
time: 0.26 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 1.39 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.39
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899482
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.39
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899423
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 1.39
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899088
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 1.39
Output dim: 0, lower bound: -0.0899066, upper bound: 0.0899066

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887873
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887873
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887910
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887910
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887871
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887871
time: 0.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898872, upper bound: 0.0898873
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898872, upper bound: 0.0898873
time: 0.25 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 1.44 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.44
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887873
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.44
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887873
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.44
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887910
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.44
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887910
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.44
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887871
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.44
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887871
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 1.44
Output dim: 0, lower bound: -0.0898872, upper bound: 0.0898873
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 1.44
Output dim: 0, lower bound: -0.0898872, upper bound: 0.0898873

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887853
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887847
time: 0.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887910
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887342
time: 0.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0880320
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0880316
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880213
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880103
time: 0.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894971, upper bound: 0.0893235
time: 0.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894787
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332
time: 0.27 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 1.55 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.55
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.55
Output dim: 0, lower bound: -0.0887843, upper bound: 0.0887853
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.55
Output dim: 0, lower bound: -0.0887332, upper bound: 0.0887873
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.55
Output dim: 0, lower bound: -0.0887828, upper bound: 0.0887847
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.55
Output dim: 0, lower bound: -0.0887862, upper bound: 0.0887910
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.55
Output dim: 0, lower bound: -0.0887871, upper bound: 0.0887342
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.55
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0880320
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.55
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0880316
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.55
Output dim: 0, lower bound: -0.0887342, upper bound: 0.0887871
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.55
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887862
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 1.55
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880213
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 1.55
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880103
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.55
Output dim: 0, lower bound: -0.0894971, upper bound: 0.0893235
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.55
Output dim: 0, lower bound: -0.0893229, upper bound: 0.0894787
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 1.55
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 1.55
Output dim: 0, lower bound: -0.0887873, upper bound: 0.0887332

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880044, upper bound: 0.0880182
time: 0.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880093
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880106
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
time: 0.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
time: 0.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
time: 0.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
time: 0.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880068, upper bound: 0.0880105
time: 0.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880093, upper bound: 0.0880198
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
time: 0.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880182, upper bound: 0.0880044
time: 0.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102
1: -0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898
2: -0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237
3: -0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035
4: -0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
time: 0.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
time: 0.24 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 1.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880044, upper bound: 0.0880182
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880198, upper bound: 0.0880093
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880216
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880122
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880106
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0880068
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880096, upper bound: 0.0880320
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880203, upper bound: 0.0880316
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880103, upper bound: 0.0879920
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880213, upper bound: 0.0879920
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880213
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0879920, upper bound: 0.0880103
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880316, upper bound: 0.0880203
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880096
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880068, upper bound: 0.0880105
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880093, upper bound: 0.0880198
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880106, upper bound: 0.0879920
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880182, upper bound: 0.0880044
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880122, upper bound: 0.0879920
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 1.53
Output dim: 0, lower bound: -0.0880216, upper bound: 0.0879920

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 1.97 + 39.50 = 41.46 seconds
