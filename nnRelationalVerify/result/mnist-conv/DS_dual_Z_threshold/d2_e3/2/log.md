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
execution time: IAR + RelationalAnalysis = 22.89 + 35.99 = 58.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.4081061, upper bound: 0.4081070

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 5734
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 470

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4081019, upper bound: 0.4069524
time: 5.11 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4069516, upper bound: 0.4081018
time: 5.25 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 10.59 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 10.59
Output dim: 0, lower bound: -0.4081019, upper bound: 0.4069524
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 10.59
Output dim: 0, lower bound: -0.4069516, upper bound: 0.4081018

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5691212, 0.5673305
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.9269583, 0.9263034
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.9020958, 0.9011450
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.7125721, 0.7126811
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8321271, 0.8341689
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7610664, 0.7606218
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8554392, 0.8550355
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7648795, 0.7628527
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.6010897, 0.6017528
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5806015, 0.5814034

Time for backsubstitution: 21.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5734
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 5734

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4080974, upper bound: 0.4037800
time: 4.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4049207, upper bound: 0.4069479
time: 5.29 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5673305, 0.5688006
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.9263031, 0.9268413
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.9011450, 0.9019244
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.7126610, 0.7125721
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8338060, 0.8321269
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7606220, 0.7609875
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8550358, 0.8553655
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7628527, 0.7645159
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.6016331, 0.6010898
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5812590, 0.5806015

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5734
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 5734

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4069470, upper bound: 0.4049207
time: 5.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4037800, upper bound: 0.4080971
time: 7.34 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 35.18 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 35.18
Output dim: 0, lower bound: -0.4080974, upper bound: 0.4037800
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 35.18
Output dim: 0, lower bound: -0.4049207, upper bound: 0.4069479
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 35.18
Output dim: 0, lower bound: -0.4069470, upper bound: 0.4049207
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 35.18
Output dim: 0, lower bound: -0.4037800, upper bound: 0.4080971

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5662428, 0.5608509
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.9177155, 0.9055188
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8927989, 0.8970094
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6927674, 0.7038622
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8251643, 0.8184967
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7598629, 0.7579229
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8450863, 0.8317652
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7393303, 0.7514613
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5978129, 0.6002917
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5781853, 0.5759754

Time for backsubstitution: 21.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 6163

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4080957, upper bound: 0.4018312
time: 4.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4061464, upper bound: 0.4037783
time: 4.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5608507, 0.5659223
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.9055190, 0.9175982
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8970094, 0.8926277
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.7038419, 0.6927674
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8181338, 0.8251643
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7579231, 0.7597842
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8317649, 0.8450136
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7514613, 0.7389667
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.6001723, 0.5978129
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5758309, 0.5781853

Time for backsubstitution: 21.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6163
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6163

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4037783, upper bound: 0.4061464
time: 6.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4018303, upper bound: 0.4080966
time: 4.11 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 32.56 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.56
Output dim: 0, lower bound: -0.4080957, upper bound: 0.4018312
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 32.56
Output dim: 0, lower bound: -0.4061464, upper bound: 0.4037783
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 32.56
Output dim: 0, lower bound: -0.4037783, upper bound: 0.4061464
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.56
Output dim: 0, lower bound: -0.4018303, upper bound: 0.4080966

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5588832, 0.5520213
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8865285, 0.8797183
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8715367, 0.8795953
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6746325, 0.6888309
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8068886, 0.8032608
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7590432, 0.7572396
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8220849, 0.8041675
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7307150, 0.7409201
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5895019, 0.5933924
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5518376, 0.5436734

Time for backsubstitution: 22.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 4657

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4080945, upper bound: 0.4015398
time: 7.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4078052, upper bound: 0.4018289
time: 7.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5520213, 0.5585628
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8797183, 0.8864105
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8795958, 0.8713663
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6888103, 0.6746325
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8028975, 0.8068888
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7572393, 0.7589648
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8041673, 0.8220122
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7409201, 0.7303514
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5932729, 0.5895019
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5435287, 0.5518376

Time for backsubstitution: 22.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 4657

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4018291, upper bound: 0.4078051
time: 8.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4015398, upper bound: 0.4080955
time: 4.71 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 36.33 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 0, lower bound: -0.4080945, upper bound: 0.4015398
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 0, lower bound: -0.4078052, upper bound: 0.4018289
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 0, lower bound: -0.4018291, upper bound: 0.4078051
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.33
Output dim: 0, lower bound: -0.4015398, upper bound: 0.4080955

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5583386, 0.5513682
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8811250, 0.8752086
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8622227, 0.8718300
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6733465, 0.6877553
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.8002944, 0.7953553
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7438183, 0.7383652
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8211889, 0.8034182
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7251065, 0.7364199
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5721892, 0.5726153
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5466900, 0.5370183

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 453

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4079664, upper bound: 0.4015396
time: 6.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4080943, upper bound: 0.4014126
time: 3.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5582302, 0.5514767
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8820186, 0.8743150
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8637714, 0.8702817
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6735570, 0.6875448
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.7989831, 0.7966664
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7401690, 0.7420142
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8213358, 0.8032711
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7262149, 0.7353115
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5687245, 0.5760797
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5451825, 0.5385258

Time for backsubstitution: 22.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 453

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4076770, upper bound: 0.4018297
time: 4.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4078050, upper bound: 0.4017019
time: 4.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5514767, 0.5579095
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8743148, 0.8819010
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8702817, 0.8636005
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6875246, 0.6735570
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.7963023, 0.7989833
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7420139, 0.7400904
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8032713, 0.8212626
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7353115, 0.7258513
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5759600, 0.5687248
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5383813, 0.5451825

Time for backsubstitution: 22.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 453

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4017010, upper bound: 0.4078059
time: 4.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4018289, upper bound: 0.4076780
time: 4.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5513682, 0.5580180
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8752084, 0.8810074
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8718300, 0.8620520
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6877351, 0.6733465
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.7949915, 0.8002944
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7383652, 0.7437394
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8034182, 0.8211157
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7364199, 0.7247429
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5724958, 0.5721892
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5368736, 0.5466900

Time for backsubstitution: 22.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 453

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4014117, upper bound: 0.4080942
time: 6.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4015395, upper bound: 0.4079664
time: 5.54 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 34.08 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 34.08
Output dim: 0, lower bound: -0.4079664, upper bound: 0.4015396
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 34.08
Output dim: 0, lower bound: -0.4080943, upper bound: 0.4014126
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 34.08
Output dim: 0, lower bound: -0.4076770, upper bound: 0.4018297
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 34.08
Output dim: 0, lower bound: -0.4078050, upper bound: 0.4017019
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 34.08
Output dim: 0, lower bound: -0.4017010, upper bound: 0.4078059
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 34.08
Output dim: 0, lower bound: -0.4018289, upper bound: 0.4076780
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 34.08
Output dim: 0, lower bound: -0.4014117, upper bound: 0.4080942
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 34.08
Output dim: 0, lower bound: -0.4015395, upper bound: 0.4079664

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5580046, 0.5512853
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8747964, 0.8736372
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8606858, 0.8655958
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6724813, 0.6842465
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.7976079, 0.7946906
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7427464, 0.7380998
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8180265, 0.8026359
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7240779, 0.7322636
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5719454, 0.5716372
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5435171, 0.5362343

Time for backsubstitution: 21.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 4582

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4077214, upper bound: 0.4015381
time: 7.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4079651, upper bound: 0.4012955
time: 3.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5582557, 0.5510341
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8795543, 0.8688798
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8559885, 0.8702931
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6698375, 0.6868901
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.7996297, 0.7926683
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7435522, 0.7372932
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8204060, 0.8002563
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7209504, 0.7353914
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5712113, 0.5723715
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5459061, 0.5338454

Time for backsubstitution: 21.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 4582

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4078494, upper bound: 0.4014112
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4080930, upper bound: 0.4011676
time: 3.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5581472, 0.5511427
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8804479, 0.8679862
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8575368, 0.8687446
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6700482, 0.6866796
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.7983184, 0.7939794
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7399035, 0.7409422
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8205528, 0.8001091
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7220588, 0.7342832
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5677466, 0.5758359
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5443983, 0.5353531

Time for backsubstitution: 22.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 4582

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4075601, upper bound: 0.4016995
time: 7.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4078037, upper bound: 0.4014569
time: 3.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5511427, 0.5578265
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8679862, 0.8803301
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8687449, 0.8573661
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6866593, 0.6700482
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.7936158, 0.7983186
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7409420, 0.7398250
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8001089, 0.8204801
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7342830, 0.7216949
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5757163, 0.5677468
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5352087, 0.5443984

Time for backsubstitution: 23.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 4582

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4014560, upper bound: 0.4078038
time: 5.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4016996, upper bound: 0.4075610
time: 3.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5510342, 0.5579350
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8688798, 0.8794363
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8702931, 0.8558176
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6868701, 0.6698375
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.7923045, 0.7996297
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7372932, 0.7434740
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8002558, 0.8203332
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7353914, 0.7205865
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5722520, 0.5712112
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5337009, 0.5459061

Time for backsubstitution: 23.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 4582

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4011667, upper bound: 0.4080938
time: 6.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4014103, upper bound: 0.4078503
time: 4.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5512853, 0.5576839
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8736367, 0.8746784
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8655958, 0.8605151
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6842263, 0.6724813
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.7943268, 0.7976074
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7381001, 0.7426674
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8026361, 0.8179538
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7322636, 0.7237141
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5715177, 0.5719454
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5360899, 0.5435172

Time for backsubstitution: 25.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 110

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 4582

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4012946, upper bound: 0.4079659
time: 5.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4015382, upper bound: 0.4077224
time: 4.29 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 35.47 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 35.47
Output dim: 0, lower bound: -0.4077214, upper bound: 0.4015381
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 35.47
Output dim: 0, lower bound: -0.4079651, upper bound: 0.4012955
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 35.47
Output dim: 0, lower bound: -0.4078494, upper bound: 0.4014112
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 35.47
Output dim: 0, lower bound: -0.4080930, upper bound: 0.4011676
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 35.47
Output dim: 0, lower bound: -0.4075601, upper bound: 0.4016995
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 35.47
Output dim: 0, lower bound: -0.4078037, upper bound: 0.4014569
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 35.47
Output dim: 0, lower bound: -0.4014560, upper bound: 0.4078038
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 35.47
Output dim: 0, lower bound: -0.4016996, upper bound: 0.4075610
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 35.47
Output dim: 0, lower bound: -0.4011667, upper bound: 0.4080938
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 35.47
Output dim: 0, lower bound: -0.4014103, upper bound: 0.4078503
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 35.47
Output dim: 0, lower bound: -0.4012946, upper bound: 0.4079659
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 35.47
Output dim: 0, lower bound: -0.4015382, upper bound: 0.4077224

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 8.0394907, 9.2233410, 8.0394907, 9.2233410, -0.5572009, 0.5508001
1: -19.7025108, -17.6858292, -19.7025108, -17.6858292, -0.8745766, 0.8732724
2: -4.7400703, -3.3977561, -4.7400703, -3.3977561, -0.8603244, 0.8653772
3: -11.3257494, -9.8784466, -11.3257494, -9.8784466, -0.6718662, 0.6832261
4: -11.1946917, -9.3216305, -11.1946917, -9.3216305, -0.7958031, 0.7936013
5: -7.2789907, -5.9051595, -7.2789907, -5.9051595, -0.7409439, 0.7370119
6: -4.2406330, -2.8002539, -4.2406330, -2.8002539, -0.8174477, 0.8022852
7: -11.7925730, -10.0609818, -11.7925730, -10.0609818, -0.7206051, 0.7265058
8: -2.8691297, -1.6222959, -2.8691297, -1.6222959, -0.5719216, 0.5715967
9: -3.7000573, -2.3618093, -3.7000573, -2.3618093, -0.5409687, 0.5320139

Time for backsubstitution: 22.05 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.88 + 562.66 = 621.54 seconds
