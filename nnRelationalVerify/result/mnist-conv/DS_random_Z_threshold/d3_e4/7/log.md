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
execution time: IAR + RelationalAnalysis = 22.87 + 35.28 = 58.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.9094677, upper bound: 0.9094670

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6123
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 5732
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 5736
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6123

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092564, upper bound: 0.9094665
time: 5.15 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094672, upper bound: 0.9092559
time: 7.84 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 13.00 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 13.00
Output dim: 0, lower bound: -0.9092564, upper bound: 0.9094665
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 13.00
Output dim: 0, lower bound: -0.9094672, upper bound: 0.9092559

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0331430, 2.0304976
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4323072, 2.4360957
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9277816, 1.9240520
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3618393, 2.3584080
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2047501, 2.2073207
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7142396, 1.7169619
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0666571, 2.0711951
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9614892, 2.9491963
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4400392, 1.4426706
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6983080, 2.6965170

Time for backsubstitution: 20.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5732
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 5736
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5732

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092230, upper bound: 0.9094660
time: 4.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092559, upper bound: 0.9094333
time: 4.70 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0304976, 2.0331430
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4360962, 2.4323072
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9240518, 1.9277813
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3584080, 2.3618393
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2073207, 2.2047501
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7169619, 1.7142396
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0711951, 2.0666566
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9491963, 2.9614892
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4426708, 1.4400392
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6965170, 2.6983080

Time for backsubstitution: 20.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 5732
type: DSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 900

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094646, upper bound: 0.9066839
time: 5.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9068953, upper bound: 0.9092531
time: 5.00 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.06 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.06
Output dim: 0, lower bound: -0.9092230, upper bound: 0.9094660
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.06
Output dim: 0, lower bound: -0.9092559, upper bound: 0.9094333
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.06
Output dim: 0, lower bound: -0.9094646, upper bound: 0.9066839
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.06
Output dim: 0, lower bound: -0.9068953, upper bound: 0.9092531

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0338378, 2.0296659
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4303751, 2.4377117
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9288654, 1.9227583
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3619118, 2.3583217
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2050080, 2.2070103
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7130370, 1.7179680
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0644336, 2.0730500
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9650998, 2.9448719
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4391766, 1.4433916
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6987534, 2.6959820

Time for backsubstitution: 19.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5736
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 5814

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5736

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9084572, upper bound: 0.9094474
time: 4.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092044, upper bound: 0.9087002
time: 6.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0323114, 2.0304976
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4323072, 2.4341626
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9264879, 1.9240520
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3617525, 2.3584080
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2044396, 2.2073207
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7142396, 1.7157593
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0666571, 2.0689721
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9571652, 2.9491963
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4400392, 1.4418080
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6977730, 2.6965170

Time for backsubstitution: 19.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 5736
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 900

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092533, upper bound: 0.9068612
time: 5.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9066840, upper bound: 0.9094332
time: 8.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0269475, 2.0283294
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4314241, 2.4288607
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9228735, 1.9256041
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3571253, 2.3608932
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2026396, 2.2012930
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7129726, 1.7112947
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0605674, 2.0522504
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9452267, 2.9561205
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4353466, 1.4301162
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6926250, 2.6954346

Time for backsubstitution: 20.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5736
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 5732

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5736

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086987, upper bound: 0.9066653
time: 6.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094460, upper bound: 0.9059180
time: 4.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0256844, 2.0295930
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4326496, 2.4276357
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9218740, 1.9266026
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3574629, 2.3605576
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2038641, 2.2000685
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7140169, 1.7102504
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0567884, 2.0560288
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9438276, 2.9575195
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4327478, 1.4327149
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6936436, 2.6944165

Time for backsubstitution: 19.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 5736
type: DSZ, layer: 1, pos: 5732
type: DSZ, layer: 1, pos: 5814

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4575

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9068873, upper bound: 0.9055838
time: 5.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9032260, upper bound: 0.9092451
time: 4.78 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.34 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 0, lower bound: -0.9084572, upper bound: 0.9094474
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 0, lower bound: -0.9092044, upper bound: 0.9087002
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 0, lower bound: -0.9092533, upper bound: 0.9068612
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 0, lower bound: -0.9066840, upper bound: 0.9094332
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 0, lower bound: -0.9086987, upper bound: 0.9066653
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 0, lower bound: -0.9094460, upper bound: 0.9059180
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 30.34
Output dim: 0, lower bound: -0.9068873, upper bound: 0.9055838
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 0, lower bound: -0.9032260, upper bound: 0.9092451

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0330625, 2.0326586
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4316463, 2.4373822
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9285088, 1.9241285
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3627315, 2.3581133
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2074342, 2.2063808
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7140923, 1.7176948
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0639954, 2.0747604
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9663606, 2.9445453
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4396808, 1.4432609
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.7007961, 2.6954565

Time for backsubstitution: 19.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5814

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9084549, upper bound: 0.9064470
time: 4.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9054567, upper bound: 0.9094458
time: 4.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0338378, 2.0288906
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4300442, 2.4377117
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9288654, 1.9224019
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3617034, 2.3583217
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2043781, 2.2070103
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7127638, 1.7179680
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0644336, 2.0726118
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9647727, 2.9448719
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4390461, 1.4433916
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6982279, 2.6959820

Time for backsubstitution: 20.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 900

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092018, upper bound: 0.9061282
time: 5.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9066325, upper bound: 0.9086974
time: 4.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0287609, 2.0256839
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4276361, 2.4307151
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9253092, 1.9218743
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3604708, 2.3574619
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.1997581, 2.2038641
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7102504, 1.7128139
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0560288, 2.0545659
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9531956, 2.9438276
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4327145, 1.4318848
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6938810, 2.6936436

Time for backsubstitution: 19.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 5736
type: DSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5745

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9085093, upper bound: 0.9068567
time: 8.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9092481, upper bound: 0.9061173
time: 4.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0274973, 2.0269475
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4288607, 2.4294901
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9243107, 1.9228733
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3608065, 2.3571262
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2009826, 2.2026396
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7112942, 1.7117701
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0522504, 2.0583444
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9517965, 2.9452267
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4301157, 1.4344835
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6948996, 2.6926250

Time for backsubstitution: 20.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5736
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5736

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9059181, upper bound: 0.9094120
time: 4.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9066653, upper bound: 0.9086671
time: 7.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0261722, 2.0313220
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4326954, 2.4285312
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9225168, 1.9269743
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3579450, 2.3606849
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2050657, 2.2006636
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7140279, 1.7110214
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0601287, 2.0539603
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9464874, 2.9557929
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4358509, 1.4299858
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6946659, 2.6949081

Time for backsubstitution: 20.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5732
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 5814

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5732

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086653, upper bound: 0.9066646
time: 5.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9086982, upper bound: 0.9066318
time: 4.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0269475, 2.0275540
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4310932, 2.4288607
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9228735, 1.9252477
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3569179, 2.3608932
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2020097, 2.2012930
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7126994, 1.7112947
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0605674, 2.0518117
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9448986, 2.9561205
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4352157, 1.4301162
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6920986, 2.6954346

Time for backsubstitution: 20.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 5732
type: DSZ, layer: 1, pos: 4575
type: DSZ, layer: 1, pos: 5745

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5814

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9094437, upper bound: 0.9048654
time: 5.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9058235, upper bound: 0.9048644
time: 5.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -1.9999886, 2.0071015
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4287834, 2.4228368
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9152904, 1.9190779
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3652658, 2.3699088
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.1903782, 2.1835299
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7171631, 1.7128782
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0451479, 2.0465302
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9240522, 2.9345942
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4215684, 1.4229343
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6894350, 2.6896067

Time for backsubstitution: 21.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5732
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 5736

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5732

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9031927, upper bound: 0.9092447
time: 4.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9032255, upper bound: 0.9092123
time: 5.05 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.27 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.27
Output dim: 0, lower bound: -0.9084549, upper bound: 0.9064470
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.27
Output dim: 0, lower bound: -0.9054567, upper bound: 0.9094458
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.27
Output dim: 0, lower bound: -0.9092018, upper bound: 0.9061282
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.27
Output dim: 0, lower bound: -0.9066325, upper bound: 0.9086974
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.27
Output dim: 0, lower bound: -0.9085093, upper bound: 0.9068567
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.27
Output dim: 0, lower bound: -0.9092481, upper bound: 0.9061173
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.27
Output dim: 0, lower bound: -0.9059181, upper bound: 0.9094120
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.27
Output dim: 0, lower bound: -0.9066653, upper bound: 0.9086671
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.27
Output dim: 0, lower bound: -0.9086653, upper bound: 0.9066646
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.27
Output dim: 0, lower bound: -0.9086982, upper bound: 0.9066318
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.27
Output dim: 0, lower bound: -0.9094437, upper bound: 0.9048654
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.27
Output dim: 0, lower bound: -0.9058235, upper bound: 0.9048644
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.27
Output dim: 0, lower bound: -0.9031927, upper bound: 0.9092447
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.27
Output dim: 0, lower bound: -0.9032255, upper bound: 0.9092123

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0187354, 2.0201955
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4183478, 2.4221849
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9285078, 1.9243593
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3688574, 2.3653374
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.1920724, 2.1884899
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7102089, 1.7124887
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0100861, 2.0275931
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9409056, 2.9233313
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4162343, 1.4227474
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6786976, 2.6702061

Time for backsubstitution: 22.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 900

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9048321, upper bound: 0.9058219
time: 4.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9048301, upper bound: 0.9094425
time: 4.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0302868, 2.0240765
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4253731, 2.4342642
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9276876, 1.9202251
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3604207, 2.3573756
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.1996970, 2.2035537
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7087746, 1.7150226
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0538058, 2.0582051
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9608016, 2.9395037
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4317219, 1.4334686
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6943359, 2.6931081

Time for backsubstitution: 21.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5814

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9091996, upper bound: 0.9050730
time: 4.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9055794, upper bound: 0.9050748
time: 5.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0290236, 2.0253401
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4265976, 2.4330387
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9266882, 1.9212241
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3607564, 2.3570399
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.2009215, 2.2023287
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7098188, 1.7139785
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0500274, 2.0619836
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9594026, 2.9409027
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4291232, 1.4360673
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6953545, 2.6920900

Time for backsubstitution: 21.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5745
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 4575

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5745

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.9058884, upper bound: 0.9086923
time: 5.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.9066274, upper bound: 0.9079535
time: 13.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 7.7540379, 10.2373848, 7.7540379, 10.2373848, -2.0279436, 2.0256839
1: -19.2597179, -15.2714062, -19.2597179, -15.2714062, -2.4276361, 2.4305639
2: -6.5238500, -3.5489047, -6.5238500, -3.5489047, -1.9230676, 1.9218743
3: -10.8192282, -7.7928081, -10.8192282, -7.7928081, -2.3570204, 2.3574619
4: -13.5905094, -10.5921221, -13.5905094, -10.5921221, -2.1997581, 2.1994433
5: -4.6404066, -2.1593924, -4.6404066, -2.1593924, -1.7102504, 1.7120080
6: -4.5149159, -1.9158846, -4.5149159, -1.9158846, -2.0560288, 2.0522256
7: -12.8235626, -8.7824345, -12.8235626, -8.7824345, -2.9472823, 2.9438276
8: -5.4501801, -3.1462450, -5.4501801, -3.1462450, -1.4327145, 1.4308500
9: -1.9316597, 1.0465155, -1.9316597, 1.0465155, -2.6935387, 2.6936436

Time for backsubstitution: 21.65 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.15 + 545.06 = 603.21 seconds
