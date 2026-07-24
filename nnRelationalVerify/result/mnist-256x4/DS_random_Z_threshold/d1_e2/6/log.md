## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01169311


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112354, 0.0112354)
1: (0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804)
2: (0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0151616, 0.0151616)
3: (-0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664)
4: (-0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922)
5: (-0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806)
6: (-0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822)
7: (-0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923)
8: (-0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544)
9: (0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.02 + 1.88 = 2.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0137566, upper bound: 0.0137566

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 221

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137531, upper bound: 0.0137152
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0137152, upper bound: 0.0137531
time: 1.08 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.47 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.47
Output dim: 9, lower bound: -0.0137531, upper bound: 0.0137152
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.47
Output dim: 9, lower bound: -0.0137152, upper bound: 0.0137531

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112148, 0.0112222
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0151146, 0.0151427
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0135016, upper bound: 0.0134600
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134847, upper bound: 0.0134780
time: 1.07 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112222, 0.0112148
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0151427, 0.0151146
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134780, upper bound: 0.0134847
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134600, upper bound: 0.0135016
time: 1.02 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.97 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.97
Output dim: 9, lower bound: -0.0135016, upper bound: 0.0134600
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.97
Output dim: 9, lower bound: -0.0134847, upper bound: 0.0134780
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.97
Output dim: 9, lower bound: -0.0134780, upper bound: 0.0134847
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.97
Output dim: 9, lower bound: -0.0134600, upper bound: 0.0135016

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112143, 0.0112216
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0151080, 0.0151337
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133477, upper bound: 0.0131634
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132242, upper bound: 0.0133067
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112148, 0.0112217
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0151056, 0.0151427
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130819, upper bound: 0.0130716
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0130819, upper bound: 0.0130716
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112217, 0.0112141
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0151363, 0.0151056
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0133060, upper bound: 0.0124230
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124559, upper bound: 0.0133181
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112222, 0.0112143
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0151337, 0.0151146
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0132904, upper bound: 0.0124565
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0124231, upper bound: 0.0133327
time: 1.02 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.06 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 9, lower bound: -0.0133477, upper bound: 0.0131634
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 9, lower bound: -0.0132242, upper bound: 0.0133067
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 9, lower bound: -0.0130819, upper bound: 0.0130716
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 9, lower bound: -0.0130819, upper bound: 0.0130716
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 9, lower bound: -0.0133060, upper bound: 0.0124230
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 9, lower bound: -0.0124559, upper bound: 0.0133181
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 9, lower bound: -0.0132904, upper bound: 0.0124565
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 9, lower bound: -0.0124231, upper bound: 0.0133327

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0111890, 0.0112095
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0150288, 0.0151022
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131811, upper bound: 0.0122078
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0122823, upper bound: 0.0129812
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112015, 0.0111963
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0150762, 0.0150546
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127907, upper bound: 0.0129187
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127907, upper bound: 0.0129187
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0111804, 0.0111965
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0149568, 0.0150231
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0107106, upper bound: 0.0107027
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0107106, upper bound: 0.0107027
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112148, 0.0111873
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0151056, 0.0149938
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129287, upper bound: 0.0127689
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0127893, upper bound: 0.0129201
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112246, 0.0112557
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0152238, 0.0153431
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0131568, upper bound: 0.0122078
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129958, upper bound: 0.0122493
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112652, 0.0112169
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0153770, 0.0151931
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120777, upper bound: 0.0129166
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0120777, upper bound: 0.0129166
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112251, 0.0112558
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0152212, 0.0153521
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129040, upper bound: 0.0120736
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0129040, upper bound: 0.0120736
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112657, 0.0112171
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0153753, 0.0152020
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0104349, upper bound: 0.0108366
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0104349, upper bound: 0.0108366
time: 0.91 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.67 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0131811, upper bound: 0.0122078
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0122823, upper bound: 0.0129812
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0127907, upper bound: 0.0129187
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0127907, upper bound: 0.0129187
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0107106, upper bound: 0.0107027
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0107106, upper bound: 0.0107027
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0129287, upper bound: 0.0127689
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0127893, upper bound: 0.0129201
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0131568, upper bound: 0.0122078
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0129958, upper bound: 0.0122493
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0120777, upper bound: 0.0129166
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0120777, upper bound: 0.0129166
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0129040, upper bound: 0.0120736
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0129040, upper bound: 0.0120736
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0104349, upper bound: 0.0108366
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.67
Output dim: 9, lower bound: -0.0104349, upper bound: 0.0108366

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0111920, 0.0112524
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0151283, 0.0153553
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105834, upper bound: 0.0101749
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105834, upper bound: 0.0101749
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112307, 0.0112125
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0152783, 0.0152017
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0101911, upper bound: 0.0105277
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0101911, upper bound: 0.0105277
time: 0.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0111670, 0.0111710
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0149273, 0.0149350
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0116886, upper bound: 0.0117949
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0116886, upper bound: 0.0117949
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112015, 0.0111618
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0150762, 0.0149057
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0104308, upper bound: 0.0104331
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0104308, upper bound: 0.0104331
time: 0.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0111895, 0.0111752
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0150265, 0.0149625
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0104428, upper bound: 0.0104181
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0104428, upper bound: 0.0104181
time: 0.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112020, 0.0111620
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0150740, 0.0149149
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0116886, upper bound: 0.0117949
time: 2.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0116886, upper bound: 0.0117949
time: 1.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0111994, 0.0112428
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0151566, 0.0153228
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105499, upper bound: 0.0101775
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105499, upper bound: 0.0101775
time: 0.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112127, 0.0112305
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0152041, 0.0152760
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125989, upper bound: 0.0118916
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0125989, upper bound: 0.0118916
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112309, 0.0111882
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0152287, 0.0150625
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0119069, upper bound: 0.0126195
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0118551, upper bound: 0.0127642
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112652, 0.0111826
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0153770, 0.0150448
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0101506, upper bound: 0.0105117
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0101506, upper bound: 0.0105117
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0111908, 0.0112279
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0150729, 0.0152273
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117845, upper bound: 0.0111246
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0117845, upper bound: 0.0111246
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112251, 0.0112215
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0152212, 0.0152037
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105028, upper bound: 0.0101486
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0105028, upper bound: 0.0101486
time: 0.75 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.51 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0105834, upper bound: 0.0101749
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0105834, upper bound: 0.0101749
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0101911, upper bound: 0.0105277
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0101911, upper bound: 0.0105277
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0116886, upper bound: 0.0117949
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0116886, upper bound: 0.0117949
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0104308, upper bound: 0.0104331
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0104308, upper bound: 0.0104331
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0104428, upper bound: 0.0104181
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0104428, upper bound: 0.0104181
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0116886, upper bound: 0.0117949
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0116886, upper bound: 0.0117949
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0105499, upper bound: 0.0101775
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0105499, upper bound: 0.0101775
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0125989, upper bound: 0.0118916
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0125989, upper bound: 0.0118916
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0119069, upper bound: 0.0126195
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0118551, upper bound: 0.0127642
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0101506, upper bound: 0.0105117
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0101506, upper bound: 0.0105117
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0117845, upper bound: 0.0111246
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0117845, upper bound: 0.0111246
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0105028, upper bound: 0.0101486
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.51
Output dim: 9, lower bound: -0.0105028, upper bound: 0.0101486

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0111443, 0.0111402
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0148516, 0.0147753
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0115196, upper bound: 0.0109555
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0109112, upper bound: 0.0116289
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0111363, 0.0111710
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0147675, 0.0149350
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0115196, upper bound: 0.0109555
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0109112, upper bound: 0.0116289
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0111792, 0.0111312
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0149975, 0.0147549
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0115196, upper bound: 0.0109555
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0109112, upper bound: 0.0116289
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0111713, 0.0111620
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0149142, 0.0149149
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0115196, upper bound: 0.0109555
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0109112, upper bound: 0.0116289
time: 1.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0111782, 0.0112025
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0150559, 0.0151512
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0115107, upper bound: 0.0109545
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0115107, upper bound: 0.0109545
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112127, 0.0111961
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0152041, 0.0151277
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0102145, upper bound: 0.0099048
time: 0.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0102145, upper bound: 0.0099048
time: 0.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112056, 0.0111744
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0151616, 0.0150364
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0099056, upper bound: 0.0102273
time: 0.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0099056, upper bound: 0.0102273
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0112181, 0.0111629
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0152088, 0.0149954
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0098955, upper bound: 0.0102407
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0098955, upper bound: 0.0102407
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0111701, 0.0111977
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0150002, 0.0150655
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0116289, upper bound: 0.0109112
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0115107, upper bound: 0.0109545
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0056372, 0.0058689, -0.0056372, 0.0058689, -0.0111606, 0.0112279
1: 0.0015671, 0.0125474, 0.0015671, 0.0125474, -0.0109804, 0.0109804
2: 0.0139266, 0.0312011, 0.0139266, 0.0312011, -0.0149113, 0.0152273
3: -0.0100901, -0.0020237, -0.0100901, -0.0020237, -0.0080664, 0.0080664
4: -0.0022926, 0.0028996, -0.0022926, 0.0028996, -0.0051922, 0.0051922
5: -0.0042298, 0.0073507, -0.0042298, 0.0073507, -0.0115806, 0.0115806
6: -0.0070884, -0.0006062, -0.0070884, -0.0006062, -0.0064822, 0.0064822
7: -0.0114622, 0.0008301, -0.0114622, 0.0008301, -0.0122923, 0.0122923
8: -0.0113532, 0.0008013, -0.0113532, 0.0008013, -0.0121544, 0.0121544
9: 0.9805366, 0.9990062, 0.9805366, 0.9990062, -0.0184696, 0.0184696

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0116289, upper bound: 0.0109112
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0115107, upper bound: 0.0109545
time: 1.11 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.96 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0115196, upper bound: 0.0109555
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0109112, upper bound: 0.0116289
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0115196, upper bound: 0.0109555
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0109112, upper bound: 0.0116289
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0115196, upper bound: 0.0109555
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0109112, upper bound: 0.0116289
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0115196, upper bound: 0.0109555
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0109112, upper bound: 0.0116289
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0115107, upper bound: 0.0109545
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0115107, upper bound: 0.0109545
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0102145, upper bound: 0.0099048
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0102145, upper bound: 0.0099048
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0099056, upper bound: 0.0102273
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0099056, upper bound: 0.0102273
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0098955, upper bound: 0.0102407
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0098955, upper bound: 0.0102407
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0116289, upper bound: 0.0109112
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0115107, upper bound: 0.0109545
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0116289, upper bound: 0.0109112
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.96
Output dim: 9, lower bound: -0.0115107, upper bound: 0.0109545

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.91 + 116.90 = 119.80 seconds
