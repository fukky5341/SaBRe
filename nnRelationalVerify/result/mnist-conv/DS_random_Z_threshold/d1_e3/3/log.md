## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.16694256000000002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4273009, 0.4273009)
1: (-11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3401191, 0.3401189)
2: (-11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4596653, 0.4596653)
3: (-10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3955762, 0.3955760)
4: (-2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2465289, 0.2465290)
5: (-9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3903337, 0.3903337)
6: (-12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3224721, 0.3224721)
7: (-6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2837343, 0.2837342)
8: (-0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3891225, 0.3891225)
9: (2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3820839, 0.3820841)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.44 + 34.52 = 58.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.1738982, upper bound: 0.1738985

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5752
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 4640

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5752

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1738905, upper bound: 0.1738981
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1738982, upper bound: 0.1738908
time: 3.31 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.46 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.46
Output dim: 9, lower bound: -0.1738905, upper bound: 0.1738981
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.46
Output dim: 9, lower bound: -0.1738982, upper bound: 0.1738908

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4242826, 0.4236786
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3338206, 0.3348711
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4467978, 0.4491053
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3805306, 0.3832512
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2443526, 0.2438140
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3686175, 0.3642681
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3183510, 0.3175268
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2816520, 0.2819984
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3888860, 0.3888395
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3821182, 0.3821130

Time for backsubstitution: 23.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 4640

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5871

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1738900, upper bound: 0.1730575
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730500, upper bound: 0.1738975
time: 3.33 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4236789, 0.4242823
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3348711, 0.3338206
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4491053, 0.4467978
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3832514, 0.3805308
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2438140, 0.2443526
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3642683, 0.3686173
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3175268, 0.3183510
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2819984, 0.2816519
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3888397, 0.3888862
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3821130, 0.3821182

Time for backsubstitution: 22.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 4640

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5871

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1738976, upper bound: 0.1730498
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730576, upper bound: 0.1738898
time: 3.15 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.61 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.61
Output dim: 9, lower bound: -0.1738900, upper bound: 0.1730575
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.61
Output dim: 9, lower bound: -0.1730500, upper bound: 0.1738975
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.61
Output dim: 9, lower bound: -0.1738976, upper bound: 0.1730498
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.61
Output dim: 9, lower bound: -0.1730576, upper bound: 0.1738898

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4138732, 0.4150057
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3323019, 0.3330505
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4454994, 0.4475236
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3799677, 0.3812587
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2406535, 0.2411196
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3643546, 0.3607144
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3166521, 0.3161113
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2774196, 0.2769194
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3792791, 0.3808353
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3797121, 0.3792269

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4640

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4640

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1735836, upper bound: 0.1730567
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1738891, upper bound: 0.1727512
time: 3.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4156094, 0.4132695
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3320000, 0.3333523
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4452162, 0.4478068
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3785381, 0.3826880
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2416582, 0.2401147
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3650637, 0.3600051
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3169358, 0.3158278
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2765727, 0.2777660
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3808818, 0.3792326
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3792324, 0.3797069

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4640

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4640

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727436, upper bound: 0.1738966
time: 5.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730491, upper bound: 0.1735912
time: 3.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4132695, 0.4156096
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3333523, 0.3320000
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4478068, 0.4452162
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3826885, 0.3785384
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2401147, 0.2416582
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3600054, 0.3650637
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3158278, 0.3169358
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2777660, 0.2765728
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3792324, 0.3808820
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3797069, 0.3792322

Time for backsubstitution: 23.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4640

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4640

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1735913, upper bound: 0.1730491
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1738968, upper bound: 0.1727435
time: 3.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4150057, 0.4138732
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3330505, 0.3323019
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4475236, 0.4454994
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3812590, 0.3799677
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2411199, 0.2406533
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3607144, 0.3643544
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3161113, 0.3166521
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2769194, 0.2774196
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3808351, 0.3792794
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3792267, 0.3797121

Time for backsubstitution: 22.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4640

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4640

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727513, upper bound: 0.1738891
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730568, upper bound: 0.1735835
time: 3.20 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.85 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.85
Output dim: 9, lower bound: -0.1735836, upper bound: 0.1730567
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.85
Output dim: 9, lower bound: -0.1738891, upper bound: 0.1727512
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.85
Output dim: 9, lower bound: -0.1727436, upper bound: 0.1738966
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.85
Output dim: 9, lower bound: -0.1730491, upper bound: 0.1735912
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.85
Output dim: 9, lower bound: -0.1735913, upper bound: 0.1730491
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.85
Output dim: 9, lower bound: -0.1738968, upper bound: 0.1727435
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.85
Output dim: 9, lower bound: -0.1727513, upper bound: 0.1738891
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.85
Output dim: 9, lower bound: -0.1730568, upper bound: 0.1735835

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4137483, 0.4147186
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3322423, 0.3329175
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4451513, 0.4473710
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3796923, 0.3811386
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2405549, 0.2408911
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3638983, 0.3605108
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3162811, 0.3152678
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2766616, 0.2765887
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3788228, 0.3797867
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3794079, 0.3790944

Time for backsubstitution: 23.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 601
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 648

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2481

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1479948, upper bound: 0.1471554
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1479948, upper bound: 0.1471554
time: 2.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4135861, 0.4148810
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3321691, 0.3329909
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4453473, 0.4471755
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3798478, 0.3809831
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2404248, 0.2410213
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3641508, 0.3602581
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3158088, 0.3157403
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2770889, 0.2761616
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3782306, 0.3803790
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3795795, 0.3789227

Time for backsubstitution: 23.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 601
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 2481

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 239

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1731055, upper bound: 0.1726588
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1737967, upper bound: 0.1719675
time: 3.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4154844, 0.4129822
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3319402, 0.3332195
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4448681, 0.4476542
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3782628, 0.3825681
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2415599, 0.2398862
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3646076, 0.3598018
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3165646, 0.3149843
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2758150, 0.2774354
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3804250, 0.3781838
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3789282, 0.3795743

Time for backsubstitution: 22.41 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 601
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3118

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1724039, upper bound: 0.1704668
time: 4.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1702103, upper bound: 0.1736210
time: 3.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4153223, 0.4131446
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3318670, 0.3332927
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4450636, 0.4474587
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3784182, 0.3824127
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2414297, 0.2400163
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3648601, 0.3595490
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3160923, 0.3154566
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2762423, 0.2770083
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3798327, 0.3787761
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3790998, 0.3794026

Time for backsubstitution: 22.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 601
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 2481

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1416

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730458, upper bound: 0.1726034
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1716095, upper bound: 0.1735894
time: 3.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4131446, 0.4153223
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3332930, 0.3318670
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4474587, 0.4450636
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3824127, 0.3784182
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2400163, 0.2414297
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3595488, 0.3648601
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3154569, 0.3160923
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2770083, 0.2762423
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3787761, 0.3798332
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3794026, 0.3790996

Time for backsubstitution: 22.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 601
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 949

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 239

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1728077, upper bound: 0.1729566
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1734989, upper bound: 0.1722654
time: 3.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4129825, 0.4154847
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3332195, 0.3319404
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4476542, 0.4448681
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3825681, 0.3782628
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2398862, 0.2415599
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3598015, 0.3646073
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3149843, 0.3165646
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2774355, 0.2758150
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3781838, 0.3804255
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3795743, 0.3789279

Time for backsubstitution: 22.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 601
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1760

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1733059, upper bound: 0.1720952
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1732470, upper bound: 0.1721215
time: 3.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4148808, 0.4135861
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3329909, 0.3321691
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4471755, 0.4453473
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3809831, 0.3798478
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2410213, 0.2404248
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3602581, 0.3641510
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3157403, 0.3158085
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2761614, 0.2770889
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3803792, 0.3782306
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3789229, 0.3795795

Time for backsubstitution: 22.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 601
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2481

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1737

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727119, upper bound: 0.1737903
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1726525, upper bound: 0.1738495
time: 4.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4147186, 0.4137483
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3329177, 0.3322423
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4473710, 0.4451513
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3811386, 0.3796923
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2408911, 0.2405550
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3605108, 0.3638983
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3152680, 0.3162811
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2765887, 0.2766618
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3797870, 0.3788228
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3790946, 0.3794079

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 601
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 2144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 423

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1704392, upper bound: 0.1708419
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1703429, upper bound: 0.1709139
time: 3.24 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.86 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.86
Output dim: 9, lower bound: -0.1479948, upper bound: 0.1471554
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.86
Output dim: 9, lower bound: -0.1479948, upper bound: 0.1471554
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.86
Output dim: 9, lower bound: -0.1731055, upper bound: 0.1726588
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.86
Output dim: 9, lower bound: -0.1737967, upper bound: 0.1719675
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.86
Output dim: 9, lower bound: -0.1724039, upper bound: 0.1704668
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.86
Output dim: 9, lower bound: -0.1702103, upper bound: 0.1736210
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.86
Output dim: 9, lower bound: -0.1730458, upper bound: 0.1726034
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.86
Output dim: 9, lower bound: -0.1716095, upper bound: 0.1735894
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.86
Output dim: 9, lower bound: -0.1728077, upper bound: 0.1729566
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.86
Output dim: 9, lower bound: -0.1734989, upper bound: 0.1722654
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.86
Output dim: 9, lower bound: -0.1733059, upper bound: 0.1720952
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.86
Output dim: 9, lower bound: -0.1732470, upper bound: 0.1721215
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.86
Output dim: 9, lower bound: -0.1727119, upper bound: 0.1737903
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.86
Output dim: 9, lower bound: -0.1726525, upper bound: 0.1738495
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.86
Output dim: 9, lower bound: -0.1704392, upper bound: 0.1708419
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.86
Output dim: 9, lower bound: -0.1703429, upper bound: 0.1709139

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4127567, 0.4143469
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3294594, 0.3307595
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4460001, 0.4482875
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3811431, 0.3823011
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2409532, 0.2416898
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3649812, 0.3608985
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3149772, 0.3150327
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2738547, 0.2734983
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3777137, 0.3799393
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3795733, 0.3790190

Time for backsubstitution: 21.52 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 601

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1416

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1731035, upper bound: 0.1712087
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1713531, upper bound: 0.1726570
time: 3.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4130518, 0.4140513
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3299377, 0.3302815
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4464593, 0.4478278
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3811655, 0.3822784
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2410934, 0.2415490
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3647914, 0.3610883
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3151011, 0.3149087
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2744257, 0.2729273
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3777909, 0.3798621
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3796763, 0.3789165

Time for backsubstitution: 21.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 601
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 716

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2481

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1479192, upper bound: 0.1463855
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1479192, upper bound: 0.1463855
time: 3.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.3651080, 0.3690724
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3328025, 0.3338170
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4207792, 0.4284966
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3745320, 0.3795004
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2143461, 0.2127964
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3637094, 0.3589237
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.2533057, 0.2589781
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2668173, 0.2690852
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3695974, 0.3658326
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3788176, 0.3794365

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 601
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2481

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1465957, upper bound: 0.1470347
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1465957, upper bound: 0.1470347
time: 2.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.3732874, 0.3626058
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3326492, 0.3340819
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4271154, 0.4235659
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3754277, 0.3788371
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2144703, 0.2135744
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3637295, 0.3591928
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.2623365, 0.2517256
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2674649, 0.2688062
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3685112, 0.3673556
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3787904, 0.3794718

Time for backsubstitution: 21.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 601
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 239

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 949

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1689969, upper bound: 0.1717072
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1682977, upper bound: 0.1724045
time: 3.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4153132, 0.4131346
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3316939, 0.3331761
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4451013, 0.4474964
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3784935, 0.3824692
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2414522, 0.2400370
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3648129, 0.3594985
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3160751, 0.3154387
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2763050, 0.2770936
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3798985, 0.3788545
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3790860, 0.3793862

Time for backsubstitution: 21.43 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.96 + 553.23 = 612.19 seconds
