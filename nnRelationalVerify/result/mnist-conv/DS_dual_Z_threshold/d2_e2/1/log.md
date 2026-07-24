## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.4356040599


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7693038, 0.7693033)
1: (-14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9576340, 0.9576340)
2: (-7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.8018274, 0.8018272)
3: (-3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8629718, 0.8629718)
4: (-8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.9078436, 0.9078436)
5: (-4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6916952, 0.6916952)
6: (-4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7937036, 0.7937036)
7: (-12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9489627, 0.9489627)
8: (6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9051771, 0.9051766)
9: (-3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.6065361, 0.6065361)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.36 + 33.41 = 55.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.4360401, upper bound: 0.4360405

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5814
type: DSZ, layer: 1, pos: 6210
type: DSZ, layer: 1, pos: 6142
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 5804
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 5814

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4355860, upper bound: 0.4360393
time: 4.03 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360389, upper bound: 0.4355860
time: 5.95 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 10.18 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 10.18
Output dim: 8, lower bound: -0.4355860, upper bound: 0.4360393
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 10.18
Output dim: 8, lower bound: -0.4360389, upper bound: 0.4355860

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7679086, 0.7647967
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9547920, 0.9581633
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7930346, 0.7901087
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8433990, 0.8482881
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.9068069, 0.9064622
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6724243, 0.6775014
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7983789, 0.7959523
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9429126, 0.9408984
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9046459, 0.9052334
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.6016161, 0.5996284

Time for backsubstitution: 21.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6210
type: DSZ, layer: 1, pos: 6142
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 5804
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 6210

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4345575, upper bound: 0.4360378
time: 4.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4355845, upper bound: 0.4350112
time: 4.14 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7647967, 0.7679083
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9581633, 0.9547920
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7901087, 0.7930346
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8482881, 0.8433990
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.9064622, 0.9068069
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6775012, 0.6724243
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7959523, 0.7983789
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9408984, 0.9429126
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9052334, 0.9046464
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5996284, 0.6016161

Time for backsubstitution: 21.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6210
type: DSZ, layer: 1, pos: 6142
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 5804
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 6210

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4350108, upper bound: 0.4355849
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360374, upper bound: 0.4345574
time: 6.29 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.37 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.37
Output dim: 8, lower bound: -0.4345575, upper bound: 0.4360378
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 31.37
Output dim: 8, lower bound: -0.4355845, upper bound: 0.4350112
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 31.37
Output dim: 8, lower bound: -0.4350108, upper bound: 0.4355849
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.37
Output dim: 8, lower bound: -0.4360374, upper bound: 0.4345574

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7656283, 0.7577915
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9544444, 0.9571042
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7917743, 0.7862668
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8423948, 0.8479571
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8994093, 0.9040556
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6719775, 0.6761341
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7979317, 0.7958050
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9386435, 0.9395103
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9032078, 0.9047623
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.6007640, 0.5970371

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6142
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 5804
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6142

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4336026, upper bound: 0.4360360
time: 5.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4345557, upper bound: 0.4350828
time: 3.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7577915, 0.7656281
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9571042, 0.9544444
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7862668, 0.7917743
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8479571, 0.8423948
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.9040556, 0.8994093
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6761341, 0.6719773
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7958050, 0.7979317
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9395103, 0.9386435
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9047623, 0.9032083
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5970371, 0.6007640

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6142
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 5804
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 6142

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4350824, upper bound: 0.4345560
time: 5.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360356, upper bound: 0.4336030
time: 4.58 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 32.11 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.11
Output dim: 8, lower bound: -0.4336026, upper bound: 0.4360360
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 32.11
Output dim: 8, lower bound: -0.4345557, upper bound: 0.4350828
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 32.11
Output dim: 8, lower bound: -0.4350824, upper bound: 0.4345560
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.11
Output dim: 8, lower bound: -0.4360356, upper bound: 0.4336030

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7596529, 0.7497475
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9451699, 0.9447403
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7906313, 0.7854908
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8452168, 0.8499470
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8965635, 0.9002652
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6692977, 0.6745467
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7909665, 0.7905779
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9368258, 0.9370894
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9019918, 0.9038491
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5985005, 0.5961909

Time for backsubstitution: 21.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 5804
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 514

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4303938, upper bound: 0.4360298
time: 4.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4335964, upper bound: 0.4328263
time: 7.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7497475, 0.7596526
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9447403, 0.9451699
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7854910, 0.7906315
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8499470, 0.8452168
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.9002652, 0.8965635
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6745467, 0.6692972
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7905779, 0.7909660
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9370894, 0.9368258
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9038486, 0.9019914
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5961909, 0.5985005

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 514
type: DSZ, layer: 1, pos: 5804
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 514

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4328262, upper bound: 0.4335964
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360293, upper bound: 0.4303937
time: 6.07 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.67 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 8, lower bound: -0.4303938, upper bound: 0.4360298
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.67
Output dim: 8, lower bound: -0.4335964, upper bound: 0.4328263
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.67
Output dim: 8, lower bound: -0.4328262, upper bound: 0.4335964
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.67
Output dim: 8, lower bound: -0.4360293, upper bound: 0.4303937

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7528806, 0.7446644
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9427390, 0.9407716
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7914500, 0.7870636
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8477221, 0.8513093
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8895721, 0.8909435
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6596551, 0.6616719
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7871852, 0.7840877
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9381285, 0.9389644
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.8912849, 0.8965597
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5865088, 0.5877571

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5804
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 5804

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4297128, upper bound: 0.4360288
time: 4.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4303928, upper bound: 0.4353487
time: 4.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7446647, 0.7528806
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9407716, 0.9427390
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7870636, 0.7914500
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8513093, 0.8477221
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8909435, 0.8895721
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6616716, 0.6596551
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7840877, 0.7871852
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9389644, 0.9381285
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.8965597, 0.8912849
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5877571, 0.5865088

Time for backsubstitution: 21.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5804
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 5804

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4353482, upper bound: 0.4303928
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360283, upper bound: 0.4297133
time: 3.86 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 29.59 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 29.59
Output dim: 8, lower bound: -0.4297128, upper bound: 0.4360288
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.59
Output dim: 8, lower bound: -0.4303928, upper bound: 0.4353487
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 29.59
Output dim: 8, lower bound: -0.4353482, upper bound: 0.4303928
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 29.59
Output dim: 8, lower bound: -0.4360283, upper bound: 0.4297133

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7461419, 0.7396097
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9383249, 0.9375877
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7879176, 0.7818718
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8409858, 0.8468604
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8829556, 0.8817711
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6522670, 0.6561286
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7856345, 0.7831087
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9315438, 0.9297318
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.8895273, 0.8956771
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5864153, 0.5874987

Time for backsubstitution: 22.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 904

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4254813, upper bound: 0.4360274
time: 6.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4297118, upper bound: 0.4317985
time: 5.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7396097, 0.7461419
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9375877, 0.9383249
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7818718, 0.7879176
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8468604, 0.8409858
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8817711, 0.8829556
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6561286, 0.6522670
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7831087, 0.7856345
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9297314, 0.9315438
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.8956766, 0.8895273
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5874987, 0.5864153

Time for backsubstitution: 21.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 904

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4317985, upper bound: 0.4297122
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360272, upper bound: 0.4254818
time: 3.95 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 29.90 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 29.90
Output dim: 8, lower bound: -0.4254813, upper bound: 0.4360274
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 29.90
Output dim: 8, lower bound: -0.4297118, upper bound: 0.4317985
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 29.90
Output dim: 8, lower bound: -0.4317985, upper bound: 0.4297122
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 29.90
Output dim: 8, lower bound: -0.4360272, upper bound: 0.4254818

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7460802, 0.7395275
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9372749, 0.9362764
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7879171, 0.7818227
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8386283, 0.8450913
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8827834, 0.8815417
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6532900, 0.6574509
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7879038, 0.7860436
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9326468, 0.9305840
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.8890409, 0.8953123
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5822761, 0.5819819

Time for backsubstitution: 22.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1702
type: DSZ, layer: 3, pos: 164
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 2824
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1970
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 911
type: DSZ, layer: 3, pos: 2615
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 2227
type: DSZ, layer: 3, pos: 2878
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 2519
type: DSZ, layer: 3, pos: 422

Time for candidate selection: 0.53 seconds

### Candidate
type: DSZ, layer: 3, pos: 1262

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4093019, upper bound: 0.4202043
time: 7.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4100959, upper bound: 0.4194116
time: 3.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7395275, 0.7460804
1: -14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9362764, 0.9372749
2: -7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.7818227, 0.7879167
3: -3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8450913, 0.8386283
4: -8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.8815417, 0.8827834
5: -4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6574509, 0.6532900
6: -4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7860436, 0.7879033
7: -12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9305840, 0.9326468
8: 6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.8953123, 0.8890409
9: -3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.5819819, 0.5822761

Time for backsubstitution: 22.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1262
type: DSZ, layer: 3, pos: 1702
type: DSZ, layer: 3, pos: 164
type: DSZ, layer: 3, pos: 2144
type: DSZ, layer: 3, pos: 1846
type: DSZ, layer: 3, pos: 2824
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1970
type: DSZ, layer: 3, pos: 418
type: DSZ, layer: 3, pos: 1248
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 911
type: DSZ, layer: 3, pos: 2615
type: DSZ, layer: 3, pos: 901
type: DSZ, layer: 3, pos: 1976
type: DSZ, layer: 3, pos: 1241
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 1849
type: DSZ, layer: 3, pos: 403
type: DSZ, layer: 3, pos: 1685
type: DSZ, layer: 3, pos: 2227
type: DSZ, layer: 3, pos: 2878
type: DSZ, layer: 3, pos: 600
type: DSZ, layer: 3, pos: 1779
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 2519
type: DSZ, layer: 3, pos: 422

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 1262

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4194115, upper bound: 0.4100959
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4202043, upper bound: 0.4093019
time: 3.12 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 28.71 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 28.71
Output dim: 8, lower bound: -0.4093019, upper bound: 0.4202043
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 28.71
Output dim: 8, lower bound: -0.4100959, upper bound: 0.4194116
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 28.71
Output dim: 8, lower bound: -0.4194115, upper bound: 0.4100959
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 28.71
Output dim: 8, lower bound: -0.4202043, upper bound: 0.4093019

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 55.77 + 390.17 = 445.94 seconds
