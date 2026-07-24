## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.566640958


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2967296, 1.2967298)
1: (-2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4835277, 1.4835272)
2: (1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2275105, 1.2275107)
3: (-6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0011253, 1.0011251)
4: (-2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9452736, 0.9452736)
5: (-4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0601525, 1.0601530)
6: (-4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4067559, 1.4067559)
7: (-8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8539648, 0.8539647)
8: (-4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4132328, 1.4132328)
9: (-11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9780154, 0.9780157)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.72 + 37.05 = 58.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.5694884, upper bound: 0.5694897

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4632
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4632

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5694867, upper bound: 0.5684073
time: 5.72 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5684076, upper bound: 0.5694867
time: 6.36 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 12.09 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 12.09
Output dim: 2, lower bound: -0.5694867, upper bound: 0.5684073
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 12.09
Output dim: 2, lower bound: -0.5684076, upper bound: 0.5694867

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2967091, 1.2967057
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4821486, 1.4824247
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2268553, 1.2267246
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0012667, 1.0012367
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9449217, 0.9449804
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0602744, 1.0603089
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4066789, 1.4067094
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8514242, 0.8518466
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4128127, 1.4127283
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9752903, 0.9757433

Time for backsubstitution: 20.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5656151, upper bound: 0.5683395
time: 4.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5694200, upper bound: 0.5645354
time: 4.10 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2967057, 1.2967088
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4824247, 1.4821486
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2267241, 1.2268555
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0012362, 1.0012670
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9449804, 0.9449217
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0603087, 1.0602741
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4067094, 1.4066789
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8518465, 0.8514242
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4127283, 1.4128127
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9757433, 0.9752903

Time for backsubstitution: 21.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 904

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5643366, upper bound: 0.5694848
time: 4.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5684055, upper bound: 0.5654139
time: 4.39 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.34 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.34
Output dim: 2, lower bound: -0.5656151, upper bound: 0.5683395
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.34
Output dim: 2, lower bound: -0.5694200, upper bound: 0.5645354
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.34
Output dim: 2, lower bound: -0.5643366, upper bound: 0.5694848
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.34
Output dim: 2, lower bound: -0.5684055, upper bound: 0.5654139

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2967143, 1.2967124
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4821510, 1.4824281
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2268467, 1.2267172
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0012841, 1.0012500
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9448907, 0.9449539
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0602968, 1.0603273
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4066691, 1.4066973
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8514280, 0.8518510
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4127965, 1.4127097
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9752829, 0.9757373

Time for backsubstitution: 20.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4625

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5654989, upper bound: 0.5653724
time: 4.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5626388, upper bound: 0.5682237
time: 4.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2967153, 1.2967112
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4821520, 1.4824271
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2268481, 1.2267160
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0012803, 1.0012541
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9448953, 0.9449494
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0602925, 1.0603313
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4066668, 1.4066997
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8514287, 0.8518503
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4127936, 1.4127121
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9752843, 0.9757359

Time for backsubstitution: 21.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 904

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5653465, upper bound: 0.5645347
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5694179, upper bound: 0.5604718
time: 4.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2906904, 1.2894876
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4635468, 1.4664187
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2202039, 1.2214212
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0024447, 1.0022268
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9392428, 0.9400017
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0706418, 1.0685554
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3936141, 1.3909664
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8581722, 0.8594933
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4116478, 1.4115531
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9733343, 0.9732249

Time for backsubstitution: 20.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 143

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5639754, upper bound: 0.5694809
time: 4.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5639769, upper bound: 0.5691202
time: 4.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2894845, 1.2906935
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4666944, 1.4632707
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2212901, 1.2203348
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0021968, 1.0024745
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9400604, 0.9391844
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0685904, 1.0706069
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3909972, 1.3935838
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8599157, 0.8577498
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4114685, 1.4117324
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9736781, 0.9728811

Time for backsubstitution: 20.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 143

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5645339, upper bound: 0.5653472
time: 4.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5683388, upper bound: 0.5615479
time: 4.34 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.20 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 29.20
Output dim: 2, lower bound: -0.5654989, upper bound: 0.5653724
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.20
Output dim: 2, lower bound: -0.5626388, upper bound: 0.5682237
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 29.20
Output dim: 2, lower bound: -0.5653465, upper bound: 0.5645347
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.20
Output dim: 2, lower bound: -0.5694179, upper bound: 0.5604718
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.20
Output dim: 2, lower bound: -0.5639754, upper bound: 0.5694809
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.20
Output dim: 2, lower bound: -0.5639769, upper bound: 0.5691202
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 29.20
Output dim: 2, lower bound: -0.5645339, upper bound: 0.5653472
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.20
Output dim: 2, lower bound: -0.5683388, upper bound: 0.5615479

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2963099, 1.2967124
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4821510, 1.4802256
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2248230, 1.2267172
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9995904, 1.0012500
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9442912, 0.9449539
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0602968, 1.0589168
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4066691, 1.4047189
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8514280, 0.8514392
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4127965, 1.4123409
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9749024, 0.9757373

Time for backsubstitution: 20.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 927

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5579784, upper bound: 0.5682219
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5626360, upper bound: 0.5635687
time: 4.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2894940, 1.2906957
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4664230, 1.4635506
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2214141, 1.2201955
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0022397, 1.0024617
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9399755, 0.9392123
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0685742, 1.0706646
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3909540, 1.3936036
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8594978, 0.8581758
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4115353, 1.4116323
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9732184, 0.9733264

Time for backsubstitution: 21.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 143

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5690537, upper bound: 0.5604659
time: 4.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5694151, upper bound: 0.5601107
time: 4.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2917066, 1.2895780
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4660397, 1.4683361
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2201643, 1.2218065
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9867220, 0.9890325
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9389644, 0.9396657
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0671892, 1.0656877
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3874009, 1.3856122
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8555453, 0.8563366
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.3961825, 1.3982482
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9727733, 0.9725506

Time for backsubstitution: 21.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 4625

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5601091, upper bound: 0.5694138
time: 4.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5639069, upper bound: 0.5656107
time: 4.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2907805, 1.2905040
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4654646, 1.4689116
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2205892, 1.2213821
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9892502, 0.9865046
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9389071, 0.9397231
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0677738, 1.0651031
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3882601, 1.3847535
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8550155, 0.8568664
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.3983431, 1.3960881
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9726598, 0.9726641

Time for backsubstitution: 21.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4625

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5638613, upper bound: 0.5661503
time: 4.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5610174, upper bound: 0.5690044
time: 6.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2894912, 1.2906988
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4666986, 1.4632745
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2212834, 1.2203264
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0022097, 1.0024920
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9400342, 0.9391537
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0686095, 1.0706301
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3909845, 1.3935730
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8599203, 0.8577535
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4114504, 1.4117167
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9736714, 0.9728734

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 143

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5679746, upper bound: 0.5615448
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5683359, upper bound: 0.5611885
time: 4.16 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.26 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 2, lower bound: -0.5579784, upper bound: 0.5682219
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.26
Output dim: 2, lower bound: -0.5626360, upper bound: 0.5635687
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 2, lower bound: -0.5690537, upper bound: 0.5604659
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 2, lower bound: -0.5694151, upper bound: 0.5601107
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 2, lower bound: -0.5601091, upper bound: 0.5694138
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.26
Output dim: 2, lower bound: -0.5639069, upper bound: 0.5656107
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.26
Output dim: 2, lower bound: -0.5638613, upper bound: 0.5661503
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 2, lower bound: -0.5610174, upper bound: 0.5690044
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 2, lower bound: -0.5679746, upper bound: 0.5615448
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 2, lower bound: -0.5683359, upper bound: 0.5611885

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.3098431, 1.3075910
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4803691, 1.4787393
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2293897, 1.2330203
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9546824, 0.9638205
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9396511, 0.9393901
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0437341, 1.0451124
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4019141, 1.3980951
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8357546, 0.8326333
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.3813624, 1.3746147
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9736876, 0.9742796

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 904

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 143

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5576118, upper bound: 0.5682186
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5579732, upper bound: 0.5678573
time: 4.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2905092, 1.2907853
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4689159, 1.4654679
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2213740, 1.2205796
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9865174, 0.9892673
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9396970, 0.9388764
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0651207, 1.0677958
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3847408, 1.3882496
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8568709, 0.8550191
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.3960705, 1.3983283
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9726582, 0.9726522

Time for backsubstitution: 22.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 4625

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 927

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5643963, upper bound: 0.5604646
time: 3.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5690513, upper bound: 0.5558209
time: 4.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2895837, 1.2917111
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4683404, 1.4660435
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2217984, 1.2201552
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9890456, 0.9867394
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9396396, 0.9389338
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0657053, 1.0672112
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3855996, 1.3873909
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8563411, 0.8555489
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.3982306, 1.3961678
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9725447, 0.9727662

Time for backsubstitution: 22.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 4625

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 927

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5647577, upper bound: 0.5601073
time: 4.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5694122, upper bound: 0.5554636
time: 4.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2917109, 1.2895837
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4660435, 1.4683404
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2201552, 1.2217984
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9867396, 0.9890456
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9389336, 0.9396396
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0672112, 1.0657055
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3873906, 1.3855999
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8555489, 0.8563411
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.3961678, 1.3982306
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9727659, 0.9725444

Time for backsubstitution: 22.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4625

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5599930, upper bound: 0.5664468
time: 4.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5571368, upper bound: 0.5692978
time: 6.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2903771, 1.2905040
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4654646, 1.4667087
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2185655, 1.2213821
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9875560, 0.9865046
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9383075, 0.9397231
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0677738, 1.0636923
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3882601, 1.3827744
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8550155, 0.8564548
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.3983431, 1.3957198
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9722793, 0.9726641

Time for backsubstitution: 21.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 927

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5563691, upper bound: 0.5690019
time: 4.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5613715, upper bound: 0.5643461
time: 4.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2905064, 1.2907882
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4691916, 1.4651923
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2212429, 1.2207105
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9864874, 0.9892976
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9397557, 0.9388177
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0651550, 1.0677609
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3847713, 1.3882186
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8572931, 0.8545969
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.3959861, 1.3984122
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9731112, 0.9721992

Time for backsubstitution: 21.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4625

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5678588, upper bound: 0.5585729
time: 4.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5650057, upper bound: 0.5614287
time: 3.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2895803, 1.2917142
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4686165, 1.4657674
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2216673, 1.2202859
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9890156, 0.9867694
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9396982, 0.9388752
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0657406, 1.0671763
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3856306, 1.3873599
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8567634, 0.8551265
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.3981462, 1.3962522
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9729977, 0.9723127

Time for backsubstitution: 20.59 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.77 + 546.68 = 605.45 seconds
