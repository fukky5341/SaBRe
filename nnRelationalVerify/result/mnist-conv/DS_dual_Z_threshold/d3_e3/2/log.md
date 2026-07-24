## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.281490264


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1638608, 1.1638610)
1: (3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5360075, 0.5360075)
2: (-4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5615702, 0.5615700)
3: (-12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.8178110, 0.8178110)
4: (-2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7643485, 0.7643486)
5: (-9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5886670, 0.5886672)
6: (-7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8692248, 0.8692250)
7: (-2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3831897, 0.3831897)
8: (-3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6626787, 0.6626787)
9: (-12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7445683, 0.7445687)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.16 + 34.84 = 57.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.2843336, upper bound: 0.2843319

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5815
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 6155
type: DSZ, layer: 1, pos: 6193
type: DSZ, layer: 1, pos: 4599

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 5815

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2794883, upper bound: 0.2794883
time: 4.18 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2794870, upper bound: 0.2843287
time: 3.49 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.93 seconds
DS_DSZ1, status: Status.VERIFIED, split count: 1, time: 7.93
Output dim: 1, lower bound: -0.2794883, upper bound: 0.2794883
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.93
Output dim: 1, lower bound: -0.2794870, upper bound: 0.2843287

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1517286, 1.1537414
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5312687, 0.5320542
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5471282, 0.5495290
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.7594726, 0.7478318
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7628214, 0.7637444
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5780988, 0.5759857
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8632191, 0.8642097
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3827448, 0.3829172
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6634128, 0.6632390
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7457840, 0.7461640

Time for backsubstitution: 21.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 6155
type: DSZ, layer: 1, pos: 6193
type: DSZ, layer: 1, pos: 4599

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 901

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2788551, upper bound: 0.2788513
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2788494, upper bound: 0.2843264
time: 3.67 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.99 seconds
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 28.99
Output dim: 1, lower bound: -0.2788551, upper bound: 0.2788513
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.99
Output dim: 1, lower bound: -0.2788494, upper bound: 0.2843264

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1477237, 1.1523013
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5313548, 0.5320539
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5449656, 0.5487524
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.7577217, 0.7429428
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7627084, 0.7626827
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5775275, 0.5743954
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8621452, 0.8638222
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3826268, 0.3834076
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6627135, 0.6629870
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7457454, 0.7460774

Time for backsubstitution: 22.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6155
type: DSZ, layer: 1, pos: 6193
type: DSZ, layer: 1, pos: 4599

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 6155

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2755130, upper bound: 0.2843181
time: 4.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2788398, upper bound: 0.2809913
time: 3.72 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.22 seconds
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.22
Output dim: 1, lower bound: -0.2755130, upper bound: 0.2843181
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 30.22
Output dim: 1, lower bound: -0.2788398, upper bound: 0.2809913

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1463823, 1.1514175
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5209168, 0.5251722
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5443034, 0.5477481
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.7501240, 0.7314212
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7550509, 0.7510724
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5393922, 0.5492868
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8355162, 0.8462802
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3763370, 0.3738626
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6477996, 0.6403575
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7239037, 0.7317085

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6193
type: DSZ, layer: 1, pos: 4599

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 6193

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2755093, upper bound: 0.2805421
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2717394, upper bound: 0.2843158
time: 3.52 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.37 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.37
Output dim: 1, lower bound: -0.2755093, upper bound: 0.2805421
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.37
Output dim: 1, lower bound: -0.2717394, upper bound: 0.2843158

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1334672, 1.1406488
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5111258, 0.5170143
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.5178438, 0.5256953
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.7587993, 0.7427127
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7604644, 0.7552288
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5443988, 0.5558029
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8232861, 0.8360810
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3575281, 0.3582774
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6655051, 0.6521970
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7153168, 0.7245491

Time for backsubstitution: 22.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4599

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 4599

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2703075, upper bound: 0.2828840
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2703061, upper bound: 0.2843087
time: 3.80 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 29.99 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 29.99
Output dim: 1, lower bound: -0.2703075, upper bound: 0.2828840
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 29.99
Output dim: 1, lower bound: -0.2703061, upper bound: 0.2843087

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1339710, 1.1412902
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5065734, 0.5115569
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.4936486, 0.5055192
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.7595673, 0.7418747
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7526696, 0.7487266
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5385845, 0.5488266
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8063095, 0.8157092
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3581936, 0.3604364
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6669680, 0.6524503
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7169471, 0.7258344

Time for backsubstitution: 21.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 662
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1971
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 325
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 1188
type: DSZ, layer: 3, pos: 698

Time for candidate selection: 0.48 seconds

### Candidate
type: DSZ, layer: 3, pos: 2578

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2598343, upper bound: 0.2723425
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2584047, upper bound: 0.2635712
time: 3.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.2602692, -10.6615553, -12.2602692, -10.6615553, -1.1341083, 1.1411524
1: 3.3816395, 4.2802763, 3.3816395, 4.2802763, -0.5056682, 0.5124619
2: -4.7594166, -3.9467888, -4.7594166, -3.9467888, -0.4976677, 0.5015000
3: -12.5689964, -11.2200060, -12.5689964, -11.2200060, -0.7579613, 0.7434802
4: -2.1814775, -1.1082797, -2.1814775, -1.1082797, -0.7539623, 0.7474339
5: -9.8950491, -8.8726807, -9.8950491, -8.8726807, -0.5374224, 0.5499886
6: -7.8550801, -6.6118288, -7.8550801, -6.6118288, -0.8029144, 0.8191046
7: -2.6614103, -2.0481133, -2.6614103, -2.0481133, -0.3596871, 0.3589429
8: -3.6533647, -2.6237564, -3.6533647, -2.6237564, -0.6657583, 0.6536598
9: -12.3033600, -11.2095757, -12.3033600, -11.2095757, -0.7166021, 0.7261796

Time for backsubstitution: 21.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2578
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1844
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 662
type: DSZ, layer: 3, pos: 961
type: DSZ, layer: 3, pos: 1971
type: DSZ, layer: 3, pos: 2321
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1984
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 325
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 1990
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 1188
type: DSZ, layer: 3, pos: 698

Time for candidate selection: 0.45 seconds

### Candidate
type: DSZ, layer: 3, pos: 2578

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2584036, upper bound: 0.2737695
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2584047, upper bound: 0.2649983
time: 3.53 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 28.74 seconds
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 28.74
Output dim: 1, lower bound: -0.2598343, upper bound: 0.2723425
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 28.74
Output dim: 1, lower bound: -0.2584047, upper bound: 0.2635712
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 28.74
Output dim: 1, lower bound: -0.2584036, upper bound: 0.2737695
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 28.74
Output dim: 1, lower bound: -0.2584047, upper bound: 0.2649983

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 57.99 + 184.18 = 242.18 seconds
