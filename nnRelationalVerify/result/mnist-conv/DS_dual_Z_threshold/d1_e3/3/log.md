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
execution time: IAR + RelationalAnalysis = 22.24 + 33.96 = 56.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.1738982, upper bound: 0.1738985

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4640
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 5752

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 4640

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1735919, upper bound: 0.1738977
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1738974, upper bound: 0.1735921
time: 4.35 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.49 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.49
Output dim: 9, lower bound: -0.1735919, upper bound: 0.1738977
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.49
Output dim: 9, lower bound: -0.1738974, upper bound: 0.1735921

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4271760, 0.4270141
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3400595, 0.3399863
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4593177, 0.4595137
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3953006, 0.3954561
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2464310, 0.2463008
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3898780, 0.3901308
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3221009, 0.3216286
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2829766, 0.2834039
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3886652, 0.3880732
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3817806, 0.3819523

Time for backsubstitution: 20.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 5752

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 5871

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1735913, upper bound: 0.1730567
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727513, upper bound: 0.1738968
time: 3.13 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4270139, 0.4271762
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3399863, 0.3400595
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4595137, 0.4593177
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3954561, 0.3953006
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2463008, 0.2464311
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3901308, 0.3898783
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3216286, 0.3221009
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2834039, 0.2829766
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3880730, 0.3886654
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3819523, 0.3817806

Time for backsubstitution: 20.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5871
type: DSZ, layer: 1, pos: 5752

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 5871

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1738968, upper bound: 0.1727512
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730568, upper bound: 0.1735912
time: 3.15 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.29 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.29
Output dim: 9, lower bound: -0.1735913, upper bound: 0.1730567
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.29
Output dim: 9, lower bound: -0.1727513, upper bound: 0.1738968
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.29
Output dim: 9, lower bound: -0.1738968, upper bound: 0.1727512
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.29
Output dim: 9, lower bound: -0.1730568, upper bound: 0.1735912

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4167662, 0.4183404
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3385403, 0.3381650
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4580178, 0.4579301
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3947382, 0.3934641
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2427318, 0.2436064
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3856144, 0.3865764
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3204017, 0.3202131
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2787440, 0.2783246
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3790584, 0.3800688
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3793750, 0.3790665

Time for backsubstitution: 20.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5752

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 5752

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1735836, upper bound: 0.1730567
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1735913, upper bound: 0.1730491
time: 3.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4185028, 0.4166040
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3382382, 0.3384669
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4577346, 0.4582138
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3933086, 0.3948934
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2437367, 0.2426016
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3863237, 0.3858671
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3206854, 0.3199294
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2778974, 0.2791712
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3806605, 0.3784659
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3788948, 0.3795464

Time for backsubstitution: 20.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5752

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 5752

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727436, upper bound: 0.1738966
time: 5.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1727513, upper bound: 0.1738891
time: 3.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4166040, 0.4185028
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3384669, 0.3382382
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4582138, 0.4577346
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3948936, 0.3933086
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2426016, 0.2437367
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3858671, 0.3863237
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3199294, 0.3206854
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2791712, 0.2778974
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3784661, 0.3806610
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3795462, 0.3788948

Time for backsubstitution: 21.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5752

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 5752

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1738891, upper bound: 0.1727512
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1738968, upper bound: 0.1727435
time: 3.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4183402, 0.4167664
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3381650, 0.3385403
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4579301, 0.4580178
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3934641, 0.3947377
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2436063, 0.2427318
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3865764, 0.3856146
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3202128, 0.3204019
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2783246, 0.2787441
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3800683, 0.3790581
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3790665, 0.3793747

Time for backsubstitution: 20.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5752

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 5752

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730491, upper bound: 0.1735912
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1730568, upper bound: 0.1735835
time: 3.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 27.16 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.16
Output dim: 9, lower bound: -0.1735836, upper bound: 0.1730567
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.16
Output dim: 9, lower bound: -0.1735913, upper bound: 0.1730491
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.16
Output dim: 9, lower bound: -0.1727436, upper bound: 0.1738966
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.16
Output dim: 9, lower bound: -0.1727513, upper bound: 0.1738891
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.16
Output dim: 9, lower bound: -0.1738891, upper bound: 0.1727512
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.16
Output dim: 9, lower bound: -0.1738968, upper bound: 0.1727435
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 27.16
Output dim: 9, lower bound: -0.1730491, upper bound: 0.1735912
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 27.16
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

Time for backsubstitution: 21.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 601

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 661

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1701052, upper bound: 0.1698989
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1704273, upper bound: 0.1695761
time: 3.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 21.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 601

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 661

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1701130, upper bound: 0.1698911
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1704349, upper bound: 0.1695678
time: 3.72 seconds

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

Time for backsubstitution: 21.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 601

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 661

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1692566, upper bound: 0.1707440
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1695813, upper bound: 0.1704243
time: 3.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 21.43 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 601

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 3, pos: 661

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1692644, upper bound: 0.1707362
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1695891, upper bound: 0.1704165
time: 3.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 21.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 601

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 661

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1704162, upper bound: 0.1695891
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1707360, upper bound: 0.1692645
time: 3.06 seconds

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

Time for backsubstitution: 21.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 601

Time for candidate selection: 0.34 seconds

### Candidate
type: DSZ, layer: 3, pos: 661

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1704240, upper bound: 0.1695813
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1707438, upper bound: 0.1692567
time: 3.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 21.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 601

Time for candidate selection: 0.39 seconds

### Candidate
type: DSZ, layer: 3, pos: 661

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1695679, upper bound: 0.1704351
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1698909, upper bound: 0.1701128
time: 3.50 seconds

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

Time for backsubstitution: 21.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 661
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 601

Time for candidate selection: 0.39 seconds

### Candidate
type: DSZ, layer: 3, pos: 661

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1695757, upper bound: 0.1704273
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1698987, upper bound: 0.1701050
time: 3.27 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.12 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.12
Output dim: 9, lower bound: -0.1701052, upper bound: 0.1698989
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.12
Output dim: 9, lower bound: -0.1704273, upper bound: 0.1695761
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.12
Output dim: 9, lower bound: -0.1701130, upper bound: 0.1698911
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.12
Output dim: 9, lower bound: -0.1704349, upper bound: 0.1695678
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.12
Output dim: 9, lower bound: -0.1692566, upper bound: 0.1707440
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.12
Output dim: 9, lower bound: -0.1695813, upper bound: 0.1704243
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.12
Output dim: 9, lower bound: -0.1692644, upper bound: 0.1707362
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.12
Output dim: 9, lower bound: -0.1695891, upper bound: 0.1704165
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.12
Output dim: 9, lower bound: -0.1704162, upper bound: 0.1695891
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.12
Output dim: 9, lower bound: -0.1707360, upper bound: 0.1692645
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.12
Output dim: 9, lower bound: -0.1704240, upper bound: 0.1695813
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.12
Output dim: 9, lower bound: -0.1707438, upper bound: 0.1692567
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.12
Output dim: 9, lower bound: -0.1695679, upper bound: 0.1704351
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.12
Output dim: 9, lower bound: -0.1698909, upper bound: 0.1701128
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.12
Output dim: 9, lower bound: -0.1695757, upper bound: 0.1704273
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.12
Output dim: 9, lower bound: -0.1698987, upper bound: 0.1701050

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4135327, 0.4144845
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3279965, 0.3284154
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4433599, 0.4460740
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3777950, 0.3800511
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2398754, 0.2397490
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3589978, 0.3503928
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3107083, 0.3122604
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2754595, 0.2756869
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3783693, 0.3770142
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3793125, 0.3789051

Time for backsubstitution: 22.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 601

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2816

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1685939, upper bound: 0.1695371
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1696499, upper bound: 0.1681425
time: 3.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4135141, 0.4145031
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3277402, 0.3286719
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4438543, 0.4455795
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3786047, 0.3792412
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2394129, 0.2402115
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3537803, 0.3556108
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3132746, 0.3096950
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2757599, 0.2753866
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3760505, 0.3793330
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3792186, 0.3789990

Time for backsubstitution: 21.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 601

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 2816

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1686723, upper bound: 0.1691201
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1700653, upper bound: 0.1680665
time: 3.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4129291, 0.4150882
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3290472, 0.3273649
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4456673, 0.4437666
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3805153, 0.3773305
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2393366, 0.2402877
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3546486, 0.3547421
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3098841, 0.3130848
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2758062, 0.2753403
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3783226, 0.3770609
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3793073, 0.3789103

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 601

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 2816

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1686020, upper bound: 0.1695293
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1696577, upper bound: 0.1681347
time: 3.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4129105, 0.4151068
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3287909, 0.3276215
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4461617, 0.4432726
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3813255, 0.3765209
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2388741, 0.2407502
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3494310, 0.3599601
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3124504, 0.3105195
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2761064, 0.2750401
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3760037, 0.3793795
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3792133, 0.3790042

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 601

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 2816

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1686802, upper bound: 0.1691123
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1700730, upper bound: 0.1680586
time: 3.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.2548418, -9.1682281, -10.2548418, -9.1682281, -0.4152689, 0.4127483
1: -11.1897335, -10.3585377, -11.1897335, -10.3585377, -0.3276947, 0.3287175
2: -11.1824360, -10.3379650, -11.1824360, -10.3379650, -0.4430766, 0.4463573
3: -10.6066313, -9.8093472, -10.6066313, -9.8093472, -0.3763654, 0.3814805
4: -2.8202238, -2.1794243, -2.8202238, -2.1794243, -0.2408801, 0.2387441
5: -9.9238567, -8.8957767, -9.9238567, -8.8957767, -0.3597071, 0.3496838
6: -12.9232407, -12.0861320, -12.9232407, -12.0861320, -0.3109918, 0.3119769
7: -6.0050840, -5.3113284, -6.0050840, -5.3113284, -0.2746129, 0.2765336
8: -0.7771082, -0.1003079, -0.7771082, -0.1003079, -0.3799720, 0.3754115
9: 2.6494007, 3.3066127, 2.6494007, 3.3066127, -0.3788328, 0.3793850

Time for backsubstitution: 21.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 675
type: DSZ, layer: 3, pos: 2481
type: DSZ, layer: 3, pos: 1235
type: DSZ, layer: 3, pos: 1737
type: DSZ, layer: 3, pos: 648
type: DSZ, layer: 3, pos: 716
type: DSZ, layer: 3, pos: 1760
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 955
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 949
type: DSZ, layer: 3, pos: 1416
type: DSZ, layer: 3, pos: 2381
type: DSZ, layer: 3, pos: 239
type: DSZ, layer: 3, pos: 601

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 2816

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1677395, upper bound: 0.1703819
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1688009, upper bound: 0.1689938
time: 3.68 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 29.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 9, lower bound: -0.1685939, upper bound: 0.1695371
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 9, lower bound: -0.1696499, upper bound: 0.1681425
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 9, lower bound: -0.1686723, upper bound: 0.1691201
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 9, lower bound: -0.1700653, upper bound: 0.1680665
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 9, lower bound: -0.1686020, upper bound: 0.1695293
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 9, lower bound: -0.1696577, upper bound: 0.1681347
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 9, lower bound: -0.1686802, upper bound: 0.1691123
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 9, lower bound: -0.1700730, upper bound: 0.1680586
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 9, lower bound: -0.1677395, upper bound: 0.1703819
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 29.22
Output dim: 9, lower bound: -0.1688009, upper bound: 0.1689938
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 9, lower bound: -0.1695813, upper bound: 0.1704243
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 9, lower bound: -0.1692644, upper bound: 0.1707362
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 9, lower bound: -0.1695891, upper bound: 0.1704165
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 9, lower bound: -0.1704162, upper bound: 0.1695891
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 9, lower bound: -0.1707360, upper bound: 0.1692645
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 9, lower bound: -0.1704240, upper bound: 0.1695813
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 9, lower bound: -0.1707438, upper bound: 0.1692567
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 9, lower bound: -0.1695679, upper bound: 0.1704351
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 9, lower bound: -0.1698909, upper bound: 0.1701128
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 9, lower bound: -0.1695757, upper bound: 0.1704273
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 9, lower bound: -0.1698987, upper bound: 0.1701050

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.20 + 547.55 = 603.75 seconds
