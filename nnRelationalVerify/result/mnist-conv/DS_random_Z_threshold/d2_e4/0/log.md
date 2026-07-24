## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.276117024


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6436245, 0.6436245)
1: (-2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7583723, 0.7583723)
2: (-3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8736706, 0.8736706)
3: (-12.2057600, -10.5023746, -12.2057600, -10.5023746, -1.0068245, 1.0068247)
4: (-5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8551059, 0.8551056)
5: (-2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7331014, 0.7331014)
6: (2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7932694, 0.7932692)
7: (-9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8936715, 0.8936715)
8: (-1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0782151, 1.0782149)
9: (-8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6931071, 0.6931071)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.67 + 33.16 = 57.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.2876219, upper bound: 0.2876191

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 430

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2876135, upper bound: 0.2807938
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2807969, upper bound: 0.2876153
time: 3.66 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.18 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.18
Output dim: 6, lower bound: -0.2876135, upper bound: 0.2807938
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.18
Output dim: 6, lower bound: -0.2807969, upper bound: 0.2876153

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6377511, 0.6402807
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7576437, 0.7570951
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8697987, 0.8714645
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9957213, 1.0005035
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8481941, 0.8429606
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7252204, 0.7192574
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7862163, 0.7808859
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8765521, 0.8839231
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0776472, 1.0778909
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6881056, 0.6902578

Time for backsubstitution: 22.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 1242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1249

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2839952, upper bound: 0.2749974
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2749990, upper bound: 0.2771785
time: 3.66 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6402805, 0.6377511
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7570951, 0.7576437
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8714647, 0.8697984
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -1.0005035, 0.9957211
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8429608, 0.8481939
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7192574, 0.7252207
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7808859, 0.7862160
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8839231, 0.8765521
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0778909, 1.0776470
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6902578, 0.6881056

Time for backsubstitution: 22.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 2495

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1977

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2772472, upper bound: 0.2868295
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2800121, upper bound: 0.2840676
time: 3.50 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.67 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.67
Output dim: 6, lower bound: -0.2839952, upper bound: 0.2749974
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.67
Output dim: 6, lower bound: -0.2749990, upper bound: 0.2771785
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.67
Output dim: 6, lower bound: -0.2772472, upper bound: 0.2868295
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.67
Output dim: 6, lower bound: -0.2800121, upper bound: 0.2840676

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.5815506, 0.5948564
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.6978252, 0.7015408
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8359756, 0.8371909
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9522145, 0.9529214
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.7953990, 0.7967267
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.6968350, 0.6843284
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7704959, 0.7599528
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8742738, 0.8809178
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0622067, 1.0628600
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6673999, 0.6729956

Time for backsubstitution: 22.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1411

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2342

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2773322, upper bound: 0.2741033
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2831036, upper bound: 0.2683274
time: 3.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.5923266, 0.5840803
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7020895, 0.6972766
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8355246, 0.8376420
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9481390, 0.9569969
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8019600, 0.7901657
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.6902916, 0.6908720
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7652831, 0.7651656
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8735468, 0.8816447
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0626159, 1.0624509
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6708431, 0.6695521

Time for backsubstitution: 22.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1824

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1389

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2811026, upper bound: 0.2756900
time: 4.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2724230, upper bound: 0.2765162
time: 3.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6402767, 0.6377461
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7583237, 0.7586527
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8529100, 0.8493178
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -1.0023575, 1.0001566
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8358819, 0.8405399
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7178960, 0.7244465
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7804067, 0.7857728
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8930130, 0.8861353
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0555077, 1.0574608
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6908236, 0.6878905

Time for backsubstitution: 23.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1727

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2495

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2752854, upper bound: 0.2834081
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2738297, upper bound: 0.2848631
time: 3.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6402757, 0.6377473
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7581041, 0.7588723
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8509841, 0.8512437
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -1.0049391, 0.9975748
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8353069, 0.8411150
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7184834, 0.7238591
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7804427, 0.7857368
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8935063, 0.8856418
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0577049, 1.0552638
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6900427, 0.6886714

Time for backsubstitution: 22.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2775779, upper bound: 0.2832907
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2792379, upper bound: 0.2816375
time: 3.58 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.64 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.64
Output dim: 6, lower bound: -0.2773322, upper bound: 0.2741033
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.64
Output dim: 6, lower bound: -0.2831036, upper bound: 0.2683274
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.64
Output dim: 6, lower bound: -0.2811026, upper bound: 0.2756900
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.64
Output dim: 6, lower bound: -0.2724230, upper bound: 0.2765162
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.64
Output dim: 6, lower bound: -0.2752854, upper bound: 0.2834081
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.64
Output dim: 6, lower bound: -0.2738297, upper bound: 0.2848631
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.64
Output dim: 6, lower bound: -0.2775779, upper bound: 0.2832907
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.64
Output dim: 6, lower bound: -0.2792379, upper bound: 0.2816375

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6050186, 0.6053050
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7522106, 0.7523718
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8470201, 0.8506613
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9932442, 0.9971395
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8369756, 0.8305728
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7038207, 0.6976986
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7405758, 0.7414889
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8798578, 0.8869996
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0509315, 1.0488286
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6445200, 0.6389194

Time for backsubstitution: 23.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 1727

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1249

### Candidate
type: DSZ, layer: 3, pos: 1780

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2758802, upper bound: 0.2737678
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2765972, upper bound: 0.2693277
time: 3.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6027753, 0.6075482
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7529204, 0.7516621
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8489947, 0.8486865
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9923568, 0.9980268
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8358059, 0.8317423
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7036617, 0.6978576
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7468193, 0.7352455
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8796287, 0.8872287
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0485845, 1.0511754
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6367671, 0.6466722

Time for backsubstitution: 23.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 1837

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2814682, upper bound: 0.2665890
time: 3.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2817769, upper bound: 0.2667352
time: 3.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.5863302, 0.5857538
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7416472, 0.7433773
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8384156, 0.8435757
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -1.0115862, 1.0195663
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8738744, 0.8628509
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7237823, 0.7166696
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7650931, 0.7608628
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8770471, 0.8844445
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0018215, 1.0170965
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6753891, 0.6769464

Time for backsubstitution: 23.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1745

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2798168, upper bound: 0.2716932
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2761266, upper bound: 0.2744261
time: 3.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.5832241, 0.5888599
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7439260, 0.7410985
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8419094, 0.8400819
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -1.0147839, 1.0163689
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8680847, 0.8686411
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7226326, 0.7178192
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7661932, 0.7597628
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8770733, 0.8844182
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0168524, 1.0020657
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6747942, 0.6775413

Time for backsubstitution: 22.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 1249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1837

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2707869, upper bound: 0.2717369
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2743349, upper bound: 0.2707265
time: 3.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6371167, 0.6367681
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7529020, 0.7532263
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8674841, 0.8674092
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9963493, 0.9888124
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8376760, 0.8442550
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7178426, 0.7232659
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7805555, 0.7856808
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8784273, 0.8735020
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0752163, 1.0731049
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6888387, 0.6867464

Time for backsubstitution: 22.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 578

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2342

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2685372, upper bound: 0.2824305
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2743939, upper bound: 0.2737749
time: 5.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6392977, 0.6345870
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7526777, 0.7534506
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8690758, 0.8658180
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9935946, 0.9915669
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8390217, 0.8429091
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7173028, 0.7238057
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7803507, 0.7858856
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8808730, 0.8710563
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0733485, 1.0749724
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6888986, 0.6866865

Time for backsubstitution: 23.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 2588

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1978

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2727922, upper bound: 0.2842162
time: 4.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2725919, upper bound: 0.2840350
time: 3.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6297026, 0.6272280
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7534378, 0.7535495
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8571057, 0.8550844
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9817083, 0.9788606
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8325896, 0.8375702
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7083008, 0.7139781
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7774155, 0.7828696
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8751619, 0.8680680
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0619900, 1.0630503
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6850171, 0.6831369

Time for backsubstitution: 23.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 1263

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2902

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2771348, upper bound: 0.2806883
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2742593, upper bound: 0.2828224
time: 3.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6297576, 0.6271729
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7530007, 0.7539864
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8567510, 0.8554389
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9836433, 0.9769258
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8323369, 0.8378229
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7080150, 0.7142639
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7775393, 0.7827458
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8754389, 0.8677909
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0632946, 1.0617464
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6852891, 0.6828647

Time for backsubstitution: 23.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 2810

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2930

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2791803, upper bound: 0.2789845
time: 4.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2765883, upper bound: 0.2815783
time: 4.35 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 32.30 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.30
Output dim: 6, lower bound: -0.2758802, upper bound: 0.2737678
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.30
Output dim: 6, lower bound: -0.2765972, upper bound: 0.2693277
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.30
Output dim: 6, lower bound: -0.2814682, upper bound: 0.2665890
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.30
Output dim: 6, lower bound: -0.2817769, upper bound: 0.2667352
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.30
Output dim: 6, lower bound: -0.2798168, upper bound: 0.2716932
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.30
Output dim: 6, lower bound: -0.2761266, upper bound: 0.2744261
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.30
Output dim: 6, lower bound: -0.2707869, upper bound: 0.2717369
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.30
Output dim: 6, lower bound: -0.2743349, upper bound: 0.2707265
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.30
Output dim: 6, lower bound: -0.2685372, upper bound: 0.2824305
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.30
Output dim: 6, lower bound: -0.2743939, upper bound: 0.2737749
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.30
Output dim: 6, lower bound: -0.2727922, upper bound: 0.2842162
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.30
Output dim: 6, lower bound: -0.2725919, upper bound: 0.2840350
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.30
Output dim: 6, lower bound: -0.2771348, upper bound: 0.2806883
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.30
Output dim: 6, lower bound: -0.2742593, upper bound: 0.2828224
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.30
Output dim: 6, lower bound: -0.2791803, upper bound: 0.2789845
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.30
Output dim: 6, lower bound: -0.2765883, upper bound: 0.2815783

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6371729, 0.6374512
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7575853, 0.7573578
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8673682, 0.8701565
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9954090, 0.9994628
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8496940, 0.8427293
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7258570, 0.7192552
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7845495, 0.7795970
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8738136, 0.8828416
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0771494, 1.0772333
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6878259, 0.6900215

Time for backsubstitution: 23.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 1242
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 1727

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2741283, upper bound: 0.2686494
time: 3.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2758230, upper bound: 0.2682998
time: 4.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6374922, 0.6394036
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7529616, 0.7541662
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8630743, 0.8650906
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9905984, 0.9947782
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8496556, 0.8421681
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7176630, 0.7099800
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7848544, 0.7808156
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8759246, 0.8830736
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0770507, 1.0771160
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6859601, 0.6887264

Time for backsubstitution: 23.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 710
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 2810

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1411

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2757686, upper bound: 0.2625999
time: 4.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.2757686, upper bound: 0.2625999
time: 4.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6368740, 0.6400218
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7547147, 0.7524130
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8634243, 0.8647408
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9899957, 0.9953811
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8474016, 0.8444223
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7159433, 0.7117000
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7861457, 0.7795243
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8757026, 0.8832955
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0768723, 1.0772946
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6865742, 0.6881123

Time for backsubstitution: 23.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1727
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2930
type: DSZ, layer: 3, pos: 1745
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 578
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 1389
type: DSZ, layer: 3, pos: 2810
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 213
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1780
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 2461
type: DSZ, layer: 3, pos: 1977
type: DSZ, layer: 3, pos: 1837
type: DSZ, layer: 3, pos: 2342
type: DSZ, layer: 3, pos: 423
type: DSZ, layer: 3, pos: 1824
type: DSZ, layer: 3, pos: 1935
type: DSZ, layer: 3, pos: 1789
type: DSZ, layer: 3, pos: 2467
type: DSZ, layer: 3, pos: 2902
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1263
type: DSZ, layer: 3, pos: 2588
type: DSZ, layer: 3, pos: 710

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1727

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2801698, upper bound: 0.2630665
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.2810537, upper bound: 0.2657974
time: 3.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.4819765, -3.4171534, -4.4819765, -3.4171534, -0.6287100, 0.6319718
1: -2.4448431, -1.2164065, -2.4448431, -1.2164065, -0.7531333, 0.7517080
2: -3.8712373, -2.7582903, -3.8712373, -2.7582903, -0.8503733, 0.8546541
3: -12.2057600, -10.5023746, -12.2057600, -10.5023746, -0.9788225, 0.9832935
4: -5.6823111, -4.6463718, -5.6823111, -4.6463718, -0.8476658, 0.8423710
5: -2.4859052, -1.3457963, -2.4859052, -1.3457963, -0.7169650, 0.7098911
6: 2.6252327, 3.5906768, 2.6252327, 3.5906768, -0.7845123, 0.7787495
7: -9.8878498, -8.5758905, -9.8878498, -8.5758905, -0.8632615, 0.8735712
8: -1.3620355, 0.3071949, -1.3620355, 0.3071949, -1.0617576, 1.0620451
9: -8.2113342, -7.3053493, -8.2113342, -7.3053493, -0.6739781, 0.6787901

Time for backsubstitution: 23.29 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.83 + 553.34 = 611.17 seconds
