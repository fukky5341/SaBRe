## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.407698893


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5688004, 0.5688006)
1: (-19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.9268413, 0.9268413)
2: (-4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.9019241, 0.9019244)
3: (-11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.7126610, 0.7126610)
4: (-11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8338060, 0.8338060)
5: (-7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7609878, 0.7609875)
6: (-4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8553658, 0.8553655)
7: (-11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7645159, 0.7645159)
8: (-2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.6016331, 0.6016332)
9: (-3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5812590, 0.5812589)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.43 + 34.73 = 58.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.4081061, upper bound: 0.4081070

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 5734
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 4582

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4569

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4080640, upper bound: 0.4081051
time: 6.69 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4081052, upper bound: 0.4080641
time: 5.69 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 12.40 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 12.40
Output dim: 0, lower bound: -0.4080640, upper bound: 0.4081051
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 12.40
Output dim: 0, lower bound: -0.4081052, upper bound: 0.4080641

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5684987, 0.5683036
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.9144258, 0.9193292
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8968358, 0.8934913
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.7098770, 0.7080479
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8264418, 0.8293602
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7583318, 0.7593813
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8495140, 0.8518310
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7613022, 0.7592051
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.6002064, 0.5992705
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5767322, 0.5785251

Time for backsubstitution: 26.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 5734

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6163

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4080623, upper bound: 0.4061540
time: 6.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4061130, upper bound: 0.4081034
time: 6.54 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5683037, 0.5684988
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.9193292, 0.9144258
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8934913, 0.8968358
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.7080479, 0.7098770
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8293605, 0.8264415
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7593818, 0.7583318
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8518310, 0.8495140
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7592051, 0.7613022
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5992706, 0.6002064
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5785251, 0.5767322

Time for backsubstitution: 22.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 5734
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 453

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6163

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4081035, upper bound: 0.4061130
time: 6.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4061542, upper bound: 0.4080621
time: 6.19 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 34.94 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 34.94
Output dim: 0, lower bound: -0.4080623, upper bound: 0.4061540
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 34.94
Output dim: 0, lower bound: -0.4061130, upper bound: 0.4081034
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 34.94
Output dim: 0, lower bound: -0.4081035, upper bound: 0.4061130
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 34.94
Output dim: 0, lower bound: -0.4061542, upper bound: 0.4080621

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5611391, 0.5594741
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8832376, 0.8935273
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8755746, 0.8760784
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6917419, 0.6930161
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8081656, 0.8141236
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7575111, 0.7586970
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8265128, 0.8242335
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7526863, 0.7486634
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5918956, 0.5923715
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5503846, 0.5462233

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 5734
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 110

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4077865, upper bound: 0.4055306
time: 4.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4074388, upper bound: 0.4058783
time: 7.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5596693, 0.5609438
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8886240, 0.8881412
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8794227, 0.8722301
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6948452, 0.6899128
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8112049, 0.8110840
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7576475, 0.7585609
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8219166, 0.8288298
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7507603, 0.7505896
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5933075, 0.5909598
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5444304, 0.5521775

Time for backsubstitution: 21.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 5734

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 837

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4061126, upper bound: 0.4074306
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4054391, upper bound: 0.4081039
time: 5.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5609438, 0.5596693
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8881409, 0.8886242
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8722301, 0.8794227
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6899128, 0.6948452
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8110838, 0.8112051
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7585611, 0.7576475
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8288298, 0.8219166
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7505894, 0.7507603
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5909595, 0.5933075
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5521775, 0.5444304

Time for backsubstitution: 21.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 5734
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 837

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 110

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4078277, upper bound: 0.4054896
time: 6.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4074801, upper bound: 0.4058372
time: 6.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5594742, 0.5611391
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8935273, 0.8832376
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8760786, 0.8755746
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6930163, 0.6917419
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8141241, 0.8081653
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7586975, 0.7575114
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8242335, 0.8265131
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7486634, 0.7526863
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5923715, 0.5918957
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5462233, 0.5503846

Time for backsubstitution: 21.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 5734

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 837

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4061538, upper bound: 0.4073883
time: 5.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4054804, upper bound: 0.4080618
time: 7.65 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 34.99 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.99
Output dim: 0, lower bound: -0.4077865, upper bound: 0.4055306
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 34.99
Output dim: 0, lower bound: -0.4074388, upper bound: 0.4058783
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 34.99
Output dim: 0, lower bound: -0.4061126, upper bound: 0.4074306
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.99
Output dim: 0, lower bound: -0.4054391, upper bound: 0.4081039
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.99
Output dim: 0, lower bound: -0.4078277, upper bound: 0.4054896
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 34.99
Output dim: 0, lower bound: -0.4074801, upper bound: 0.4058372
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 34.99
Output dim: 0, lower bound: -0.4061538, upper bound: 0.4073883
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.99
Output dim: 0, lower bound: -0.4054804, upper bound: 0.4080618

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5611382, 0.5594738
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8832395, 0.8935289
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8755736, 0.8760769
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6917427, 0.6930170
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8081656, 0.8141236
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7575111, 0.7586963
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8265128, 0.8242331
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7526844, 0.7486620
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5918951, 0.5923712
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5503836, 0.5462224

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 5734
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 4657

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 453

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4076584, upper bound: 0.4055314
time: 7.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4077863, upper bound: 0.4054036
time: 3.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5593357, 0.5609028
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8891068, 0.8883920
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8801508, 0.8730993
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6930325, 0.6872458
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8113246, 0.8110838
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7581584, 0.7588642
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8199832, 0.8260984
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7508307, 0.7505894
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5927521, 0.5902929
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5443652, 0.5522063

Time for backsubstitution: 22.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 5734

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 110

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4051634, upper bound: 0.4074798
time: 5.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4048157, upper bound: 0.4078273
time: 5.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5609426, 0.5596689
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8881433, 0.8886259
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8722291, 0.8794212
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6899137, 0.6948459
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8110838, 0.8112051
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7585607, 0.7576468
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8288298, 0.8219161
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7505877, 0.7507589
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5909591, 0.5933073
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5521765, 0.5444295

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 5734
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4657

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4582

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4075828, upper bound: 0.4054890
time: 5.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4078265, upper bound: 0.4052444
time: 7.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5591404, 0.5610981
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8940105, 0.8834884
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8768063, 0.8764439
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6912036, 0.6890748
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8142428, 0.8081653
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7592084, 0.7578146
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8223002, 0.8237815
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7487338, 0.7526863
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5918161, 0.5912290
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5461581, 0.5504134

Time for backsubstitution: 21.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5734
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5734

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4054758, upper bound: 0.4048805
time: 5.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4022992, upper bound: 0.4080582
time: 5.54 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 32.79 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.79
Output dim: 0, lower bound: -0.4076584, upper bound: 0.4055314
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.79
Output dim: 0, lower bound: -0.4077863, upper bound: 0.4054036
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.79
Output dim: 0, lower bound: -0.4051634, upper bound: 0.4074798
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.79
Output dim: 0, lower bound: -0.4048157, upper bound: 0.4078273
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.79
Output dim: 0, lower bound: -0.4075828, upper bound: 0.4054890
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.79
Output dim: 0, lower bound: -0.4078265, upper bound: 0.4052444
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.79
Output dim: 0, lower bound: -0.4054758, upper bound: 0.4048805
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.79
Output dim: 0, lower bound: -0.4022992, upper bound: 0.4080582

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5610547, 0.5591394
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8816693, 0.8872008
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8693385, 0.8745396
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6882334, 0.6921511
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8075018, 0.8114381
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7572465, 0.7576249
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8257298, 0.8210709
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7485278, 0.7476332
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5909171, 0.5921276
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5495994, 0.5430496

Time for backsubstitution: 21.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 5734
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 4657

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4582

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4075413, upper bound: 0.4054021
time: 10.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4077851, upper bound: 0.4051576
time: 5.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5593352, 0.5609018
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8891087, 0.8883939
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8801498, 0.8730993
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6930330, 0.6872463
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8113236, 0.8110833
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7581575, 0.7588639
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8199818, 0.8260970
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7508292, 0.7505875
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5927517, 0.5902921
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5443640, 0.5522048

Time for backsubstitution: 21.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5734
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4657

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5734

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4048112, upper bound: 0.4046471
time: 5.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4016346, upper bound: 0.4078228
time: 6.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5604582, 0.5588658
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8877778, 0.8884056
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8720112, 0.8790605
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6888931, 0.6942296
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8099947, 0.8093996
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7574725, 0.7558441
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8284774, 0.8213363
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7448292, 0.7472854
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5909183, 0.5932827
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5479562, 0.5418825

Time for backsubstitution: 21.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 5734
type: DSZ, layer: 1, pos: 453

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 470

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4078224, upper bound: 0.4040928
time: 4.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4066726, upper bound: 0.4052401
time: 4.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5526607, 0.5582196
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8732259, 0.8742456
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8726702, 0.8671470
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6823843, 0.6692696
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.7985697, 0.8012021
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7565100, 0.7566121
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.7990303, 0.8134294
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7373428, 0.7271373
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5903548, 0.5879519
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5407299, 0.5479969

Time for backsubstitution: 21.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 110
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 4657

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 453

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4021711, upper bound: 0.4080569
time: 5.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4022990, upper bound: 0.4079291
time: 6.73 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 34.02 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 34.02
Output dim: 0, lower bound: -0.4075413, upper bound: 0.4054021
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 34.02
Output dim: 0, lower bound: -0.4077851, upper bound: 0.4051576
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 34.02
Output dim: 0, lower bound: -0.4048112, upper bound: 0.4046471
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 34.02
Output dim: 0, lower bound: -0.4016346, upper bound: 0.4078228
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 34.02
Output dim: 0, lower bound: -0.4078224, upper bound: 0.4040928
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 34.02
Output dim: 0, lower bound: -0.4066726, upper bound: 0.4052401
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 34.02
Output dim: 0, lower bound: -0.4021711, upper bound: 0.4080569
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 34.02
Output dim: 0, lower bound: -0.4022990, upper bound: 0.4079291

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5605707, 0.5583360
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8813043, 0.8869810
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8691211, 0.8741786
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6872127, 0.6915352
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8064127, 0.8096325
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7561579, 0.7558222
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8253784, 0.8204908
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7427690, 0.7441609
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5908759, 0.5921029
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5453792, 0.5405025

Time for backsubstitution: 21.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5734
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 470

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5734

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4077805, upper bound: 0.4019771
time: 13.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4046039, upper bound: 0.4051531
time: 6.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5528555, 0.5580233
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8683240, 0.8791511
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8760128, 0.8638017
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6842136, 0.6674409
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.7956514, 0.8041208
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7554588, 0.7576611
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.7967119, 0.8157449
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7394378, 0.7250385
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5912910, 0.5870154
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5389359, 0.5497885

Time for backsubstitution: 21.47 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.16 + 552.76 = 610.91 seconds
