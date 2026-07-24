## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.9085582323


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0339298, 2.0339298)
1: (-19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4372330, 2.4372330)
2: (-6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9289122, 1.9289124)
3: (-10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3628707, 2.3628707)
4: (-13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2080922, 2.2080922)
5: (-4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7177835, 1.7177835)
6: (-4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0725527, 2.0725527)
7: (-12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9651766, 2.9651771)
8: (-5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4434562, 1.4434562)
9: (-1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6988459, 2.6988463)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.00 + 34.31 = 57.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.9094677, upper bound: 0.9094670

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5736
type: DSZ, layer: 1, pos: 5732
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5736

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9087018, upper bound: 0.9094483
time: 5.33 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094490, upper bound: 0.9087012
time: 5.74 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 11.16 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 11.16
Output dim: 0, lower bound: -0.9087018, upper bound: 0.9094483
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 11.16
Output dim: 0, lower bound: -0.9094490, upper bound: 0.9087012

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0331545, 2.0369225
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4385042, 2.4369030
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9285564, 1.9302831
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3636904, 2.3626623
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2105188, 2.2074628
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7188382, 1.7175102
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0721140, 2.0742626
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9664373, 2.9648490
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4439600, 1.4433253
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.7008896, 2.6983213

Time for backsubstitution: 21.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5732
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 5732

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086684, upper bound: 0.9094478
time: 5.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9087012, upper bound: 0.9094151
time: 7.14 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0339298, 2.0331545
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4369030, 2.4372330
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9289122, 1.9285560
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3626623, 2.3628707
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2074628, 2.2080922
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7175107, 1.7177835
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0725527, 2.0721140
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9648495, 2.9651771
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4433258, 1.4434562
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6983213, 2.6988463

Time for backsubstitution: 21.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5732
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5732

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094156, upper bound: 0.9087006
time: 6.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094484, upper bound: 0.9086678
time: 6.07 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 34.20 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 34.20
Output dim: 0, lower bound: -0.9086684, upper bound: 0.9094478
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 34.20
Output dim: 0, lower bound: -0.9087012, upper bound: 0.9094151
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 34.20
Output dim: 0, lower bound: -0.9094156, upper bound: 0.9087006
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 34.20
Output dim: 0, lower bound: -0.9094484, upper bound: 0.9086678

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0338492, 2.0360909
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4365711, 2.4385195
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9296398, 1.9289894
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3637629, 2.3625755
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2107763, 2.2071524
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7176361, 1.7185163
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0698915, 2.0761209
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9700484, 2.9605246
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4430978, 1.4440455
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.7013369, 2.6977882

Time for backsubstitution: 21.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 5814

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086662, upper bound: 0.9064475
time: 4.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9056680, upper bound: 0.9094456
time: 4.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0323234, 2.0369225
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4385042, 2.4349689
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9272633, 1.9302831
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3636045, 2.3626623
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2102079, 2.2074628
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7188382, 1.7163084
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0721140, 2.0720401
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9621129, 2.9648490
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4439600, 1.4424627
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.7003565, 2.6983213

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5814

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086990, upper bound: 0.9064147
time: 4.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9057008, upper bound: 0.9094128
time: 4.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0346246, 2.0323229
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4349689, 2.4388494
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9299965, 1.9272628
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3627348, 2.3627849
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2077208, 2.2077813
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7163086, 1.7187893
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0703297, 2.0739717
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9684606, 2.9608517
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4424627, 1.4441762
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6987686, 2.6983118

Time for backsubstitution: 21.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 5814

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094134, upper bound: 0.9057001
time: 5.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9064152, upper bound: 0.9086984
time: 5.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0330987, 2.0331545
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4369030, 2.4352989
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9276190, 1.9285560
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3625765, 2.3628707
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2071524, 2.2080922
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7175107, 1.7165813
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0725527, 2.0698915
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9605250, 2.9651771
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4433258, 1.4425933
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6977882, 2.6988463

Time for backsubstitution: 21.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 5814

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094462, upper bound: 0.9056674
time: 5.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9064480, upper bound: 0.9086656
time: 6.06 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 33.28 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.28
Output dim: 0, lower bound: -0.9086662, upper bound: 0.9064475
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.28
Output dim: 0, lower bound: -0.9056680, upper bound: 0.9094456
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.28
Output dim: 0, lower bound: -0.9086990, upper bound: 0.9064147
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.28
Output dim: 0, lower bound: -0.9057008, upper bound: 0.9094128
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.28
Output dim: 0, lower bound: -0.9094134, upper bound: 0.9057001
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.28
Output dim: 0, lower bound: -0.9064152, upper bound: 0.9086984
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.28
Output dim: 0, lower bound: -0.9094462, upper bound: 0.9056674
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.28
Output dim: 0, lower bound: -0.9064480, upper bound: 0.9086656

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0213866, 2.0217643
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4213748, 2.4252214
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9298697, 1.9289875
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3709874, 2.3687038
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.1928859, 2.1917906
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7124300, 1.7146330
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0227232, 2.0222106
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9488344, 2.9350681
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4225838, 1.4205990
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6760836, 2.6756868

Time for backsubstitution: 22.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4575

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086582, upper bound: 0.9027793
time: 4.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9049969, upper bound: 0.9064395
time: 4.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0195227, 2.0236282
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4232726, 2.4233232
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9296389, 1.9292192
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3698907, 2.3698010
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.1954150, 2.1892614
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7137532, 1.7133102
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0159817, 2.0289521
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9445925, 2.9393101
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4196513, 1.4235318
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6792355, 2.6725349

Time for backsubstitution: 22.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 4575

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9056599, upper bound: 0.9057773
time: 4.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9019987, upper bound: 0.9094378
time: 4.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0198603, 2.0225964
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4233088, 2.4216709
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9274931, 1.9302807
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3708291, 2.3687892
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.1923175, 2.1921000
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7136316, 1.7124250
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0249462, 2.0181303
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9408979, 2.9393940
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4234469, 1.4190161
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6751032, 2.6762214

Time for backsubstitution: 21.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4575

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086910, upper bound: 0.9027462
time: 4.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9050297, upper bound: 0.9064067
time: 4.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0179963, 2.0244598
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4252067, 2.4197726
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9272604, 1.9305129
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3697314, 2.3698869
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.1948466, 2.1895709
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7149544, 1.7111020
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0182047, 2.0248718
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9366570, 2.9436350
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4205139, 1.4219489
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6782551, 2.6730690

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 4575

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9056928, upper bound: 0.9057446
time: 4.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9020315, upper bound: 0.9094050
time: 5.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0221620, 2.0179963
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4197726, 2.4255505
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9302263, 1.9272609
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3699594, 2.3689117
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.1898298, 2.1924195
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7111020, 1.7149067
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0231624, 2.0200620
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9472456, 2.9353962
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4219491, 1.4207299
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6735153, 2.6762123

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4575

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094055, upper bound: 0.9020310
time: 4.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9057450, upper bound: 0.9056924
time: 4.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0202980, 2.0198603
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4216704, 2.4236522
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9299946, 1.9274931
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3688626, 2.3700094
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.1923594, 2.1898904
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7124252, 1.7135835
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0164208, 2.0268035
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9430046, 2.9396377
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4190161, 1.4236627
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6766682, 2.6730604

Time for backsubstitution: 21.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 4575

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9064073, upper bound: 0.9050291
time: 4.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9027468, upper bound: 0.9086903
time: 5.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0206356, 2.0188284
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4217067, 2.4219999
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9278498, 1.9285541
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3698010, 2.3689981
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.1892614, 2.1927299
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7123032, 1.7126985
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0253849, 2.0159812
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9393101, 2.9397216
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4228117, 1.4191470
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6725349, 2.6767468

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4575

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094383, upper bound: 0.9019981
time: 4.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9057778, upper bound: 0.9056594
time: 4.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0187716, 2.0206919
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4236045, 2.4201016
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9276171, 1.9287863
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3687034, 2.3700957
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.1917906, 2.1902003
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7136259, 1.7113757
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0186434, 2.0227232
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9350681, 2.9439631
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4198792, 1.4220800
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6756868, 2.6735945

Time for backsubstitution: 22.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 4575

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9064401, upper bound: 0.9049963
time: 5.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9027796, upper bound: 0.9086577
time: 6.04 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 34.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.13
Output dim: 0, lower bound: -0.9086582, upper bound: 0.9027793
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.13
Output dim: 0, lower bound: -0.9049969, upper bound: 0.9064395
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 34.13
Output dim: 0, lower bound: -0.9056599, upper bound: 0.9057773
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.13
Output dim: 0, lower bound: -0.9019987, upper bound: 0.9094378
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.13
Output dim: 0, lower bound: -0.9086910, upper bound: 0.9027462
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.13
Output dim: 0, lower bound: -0.9050297, upper bound: 0.9064067
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 34.13
Output dim: 0, lower bound: -0.9056928, upper bound: 0.9057446
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.13
Output dim: 0, lower bound: -0.9020315, upper bound: 0.9094050
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.13
Output dim: 0, lower bound: -0.9094055, upper bound: 0.9020310
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.13
Output dim: 0, lower bound: -0.9057450, upper bound: 0.9056924
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 34.13
Output dim: 0, lower bound: -0.9064073, upper bound: 0.9050291
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.13
Output dim: 0, lower bound: -0.9027468, upper bound: 0.9086903
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.13
Output dim: 0, lower bound: -0.9094383, upper bound: 0.9019981
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.13
Output dim: 0, lower bound: -0.9057778, upper bound: 0.9056594
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 34.13
Output dim: 0, lower bound: -0.9064401, upper bound: 0.9049963
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.13
Output dim: 0, lower bound: -0.9027796, upper bound: 0.9086577

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -1.9988942, 1.9960680
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4165754, 2.4213557
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9223461, 1.9224055
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3803368, 2.3765059
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.1763482, 2.1783061
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7150578, 1.7177792
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0132251, 2.0105705
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9259081, 2.9152927
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4128036, 1.4094203
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6712742, 2.6714802

Time for backsubstitution: 22.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 900

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086556, upper bound: 0.9021524
time: 4.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9050351, upper bound: 0.9021544
time: 4.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -1.9938264, 2.0011363
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4194059, 2.4185243
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9230556, 1.9216955
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3776922, 2.3791509
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.1819301, 2.1727242
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7168994, 1.7159376
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0043416, 2.0194545
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9248171, 2.9163837
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4084730, 1.4137518
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6750278, 2.6677256

Time for backsubstitution: 22.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 900

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9013741, upper bound: 0.9058145
time: 5.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9013721, upper bound: 0.9094351
time: 4.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -1.9973683, 1.9968991
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4185076, 2.4178052
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9199686, 1.9236982
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3801785, 2.3765922
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.1757798, 2.1786156
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7162600, 1.7155712
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0154471, 2.0064902
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9179726, 2.9196177
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4136667, 1.4078374
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6702938, 2.6720142

Time for backsubstitution: 23.41 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.31 + 554.07 = 611.38 seconds
