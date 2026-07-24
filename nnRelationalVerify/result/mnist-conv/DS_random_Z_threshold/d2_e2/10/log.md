## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.287744645


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6356344, 0.6356342)
1: (-7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5325336, 0.5325336)
2: (-7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5109730, 0.5109732)
3: (-5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7120657, 0.7120652)
4: (-7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6489444, 0.6489446)
5: (-0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5138854, 0.5138855)
6: (-2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6067991, 0.6067991)
7: (-10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6091228, 0.6091225)
8: (7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4724457, 0.4724457)
9: (-5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7989490, 0.7989488)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.45 + 34.45 = 58.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.3028891, upper bound: 0.3028899

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 4596

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3025187, upper bound: 0.3025293
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3025286, upper bound: 0.3025188
time: 4.23 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.96 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.96
Output dim: 8, lower bound: -0.3025187, upper bound: 0.3025293
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.96
Output dim: 8, lower bound: -0.3025286, upper bound: 0.3025188

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6356335, 0.6356349
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5325336, 0.5325339
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5109730, 0.5109742
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7120647, 0.7120628
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6489463, 0.6489437
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5138862, 0.5138855
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6067986, 0.6068003
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6091228, 0.6091225
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4724455, 0.4724457
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7989500, 0.7989483

Time for backsubstitution: 22.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 4596

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 568

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3025185, upper bound: 0.3025226
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3025120, upper bound: 0.3025291
time: 4.25 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6356344, 0.6356335
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5325336, 0.5325336
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5109730, 0.5109730
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7120657, 0.7120647
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6489434, 0.6489446
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5138853, 0.5138855
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6067991, 0.6067986
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6091228, 0.6091225
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4724457, 0.4724456
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7989480, 0.7989488

Time for backsubstitution: 22.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 4596

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3024960, upper bound: 0.3024928
time: 4.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3025282, upper bound: 0.3024952
time: 3.84 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.04 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.04
Output dim: 8, lower bound: -0.3025185, upper bound: 0.3025226
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.04
Output dim: 8, lower bound: -0.3025120, upper bound: 0.3025291
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.04
Output dim: 8, lower bound: -0.3024960, upper bound: 0.3024928
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.04
Output dim: 8, lower bound: -0.3025282, upper bound: 0.3024952

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6362734, 0.6353071
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5325346, 0.5325327
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5109520, 0.5110166
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7131100, 0.7115278
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6483116, 0.6501801
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5138600, 0.5139382
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6068058, 0.6067960
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6086688, 0.6100054
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4724615, 0.4724382
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7992666, 0.7987864

Time for backsubstitution: 23.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4596
type: DSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4596

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016486, upper bound: 0.3025195
time: 4.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3025162, upper bound: 0.3016531
time: 4.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6353054, 0.6356349
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5325327, 0.5325339
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5109730, 0.5109532
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7115297, 0.7120628
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6489463, 0.6483090
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5138862, 0.5138590
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6067944, 0.6068003
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6091228, 0.6086686
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4724381, 0.4724457
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7987878, 0.7989483

Time for backsubstitution: 23.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4596
type: DSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4596

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016421, upper bound: 0.3025268
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3025097, upper bound: 0.3016588
time: 5.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6298800, 0.6340289
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5319109, 0.5303061
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5097606, 0.5106342
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7113442, 0.7094822
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6478224, 0.6449246
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5134712, 0.5123971
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6002579, 0.6049707
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6075859, 0.6036072
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4719982, 0.4723216
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7981987, 0.7962577

Time for backsubstitution: 23.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 4596

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 568

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3024958, upper bound: 0.3024861
time: 5.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3024893, upper bound: 0.3024921
time: 4.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6340289, 0.6298802
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5303063, 0.5319109
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5106347, 0.5097604
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7094827, 0.7113438
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6449242, 0.6478231
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5123969, 0.5134711
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6049709, 0.6002576
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6036072, 0.6075859
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4723217, 0.4719981
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7962575, 0.7981987

Time for backsubstitution: 23.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4596
type: DSZ, layer: 1, pos: 568

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4596

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016587, upper bound: 0.3024928
time: 4.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3025259, upper bound: 0.3016253
time: 4.10 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.58 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.58
Output dim: 8, lower bound: -0.3016486, upper bound: 0.3025195
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.58
Output dim: 8, lower bound: -0.3025162, upper bound: 0.3016531
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.58
Output dim: 8, lower bound: -0.3016421, upper bound: 0.3025268
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.58
Output dim: 8, lower bound: -0.3025097, upper bound: 0.3016588
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.58
Output dim: 8, lower bound: -0.3024958, upper bound: 0.3024861
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.58
Output dim: 8, lower bound: -0.3024893, upper bound: 0.3024921
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.58
Output dim: 8, lower bound: -0.3016587, upper bound: 0.3024928
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.58
Output dim: 8, lower bound: -0.3025259, upper bound: 0.3016253

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6312099, 0.6315081
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5241466, 0.5263779
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5085592, 0.5096972
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7079115, 0.7077489
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6435328, 0.6438091
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5106096, 0.5094547
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6068535, 0.6068318
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6067700, 0.6091168
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4694676, 0.4702804
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7951565, 0.7931721

Time for backsubstitution: 22.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016244, upper bound: 0.3025199
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016221, upper bound: 0.3024877
time: 5.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6324744, 0.6302438
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5263801, 0.5241449
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5096321, 0.5086238
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7093306, 0.7063293
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6419401, 0.6454012
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5093765, 0.5106883
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6068416, 0.6068435
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6077800, 0.6081069
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4703035, 0.4694444
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7936521, 0.7946763

Time for backsubstitution: 22.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3024919, upper bound: 0.3016527
time: 3.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3024895, upper bound: 0.3016201
time: 3.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6302423, 0.6318359
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5241446, 0.5263791
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5085802, 0.5096335
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7063313, 0.7082844
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6441669, 0.6419377
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5106368, 0.5093756
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6068420, 0.6068361
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6072240, 0.6077797
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4694443, 0.4702879
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7946777, 0.7933352

Time for backsubstitution: 23.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016178, upper bound: 0.3025264
time: 5.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016156, upper bound: 0.3024942
time: 3.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6315064, 0.6305716
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5263782, 0.5241461
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5096536, 0.5085604
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7077503, 0.7068653
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6425743, 0.6435299
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5094032, 0.5106091
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6068301, 0.6068478
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6082335, 0.6067703
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4702802, 0.4694519
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7931738, 0.7948394

Time for backsubstitution: 23.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3024854, upper bound: 0.3016592
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3024830, upper bound: 0.3016267
time: 3.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6305194, 0.6337013
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5319118, 0.5303051
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5097392, 0.5106764
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7123895, 0.7089467
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6471882, 0.6461613
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5134448, 0.5124495
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6002655, 0.6049664
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6071320, 0.6044900
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4720140, 0.4723141
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7985146, 0.7960951

Time for backsubstitution: 23.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4596

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4596

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016259, upper bound: 0.3024838
time: 4.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3024935, upper bound: 0.3016163
time: 3.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6295524, 0.6340289
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5319099, 0.5303061
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5097606, 0.5106130
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7108092, 0.7094822
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6478224, 0.6442900
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5134712, 0.5123703
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6002536, 0.6049707
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6075859, 0.6031532
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4719906, 0.4723216
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7980359, 0.7962577

Time for backsubstitution: 23.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4596

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4596

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016194, upper bound: 0.3024903
time: 4.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3024870, upper bound: 0.3016228
time: 3.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6289659, 0.6260815
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5219183, 0.5257568
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5082426, 0.5084412
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7042847, 0.7075644
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6401453, 0.6414521
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5091481, 0.5089889
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6050200, 0.6002932
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6017089, 0.6066978
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4693277, 0.4698400
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7921481, 0.7925851

Time for backsubstitution: 23.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 568

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 568

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016585, upper bound: 0.3024861
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3016520, upper bound: 0.3024926
time: 4.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6302304, 0.6248171
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5241513, 0.5235231
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5093160, 0.5073678
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7057037, 0.7061458
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6385527, 0.6430438
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5079145, 0.5102220
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6050072, 0.6003051
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6027188, 0.6056881
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4701638, 0.4690040
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7906437, 0.7940893

Time for backsubstitution: 28.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 568

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 568

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3025257, upper bound: 0.3016179
time: 4.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3025192, upper bound: 0.3016251
time: 3.46 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 36.38 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.38
Output dim: 8, lower bound: -0.3016244, upper bound: 0.3025199
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.38
Output dim: 8, lower bound: -0.3016221, upper bound: 0.3024877
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.38
Output dim: 8, lower bound: -0.3024919, upper bound: 0.3016527
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.38
Output dim: 8, lower bound: -0.3024895, upper bound: 0.3016201
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.38
Output dim: 8, lower bound: -0.3016178, upper bound: 0.3025264
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.38
Output dim: 8, lower bound: -0.3016156, upper bound: 0.3024942
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.38
Output dim: 8, lower bound: -0.3024854, upper bound: 0.3016592
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.38
Output dim: 8, lower bound: -0.3024830, upper bound: 0.3016267
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.38
Output dim: 8, lower bound: -0.3016259, upper bound: 0.3024838
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.38
Output dim: 8, lower bound: -0.3024935, upper bound: 0.3016163
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.38
Output dim: 8, lower bound: -0.3016194, upper bound: 0.3024903
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.38
Output dim: 8, lower bound: -0.3024870, upper bound: 0.3016228
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.38
Output dim: 8, lower bound: -0.3016585, upper bound: 0.3024861
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.38
Output dim: 8, lower bound: -0.3016520, upper bound: 0.3024926
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.38
Output dim: 8, lower bound: -0.3025257, upper bound: 0.3016179
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.38
Output dim: 8, lower bound: -0.3025192, upper bound: 0.3016251

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6254568, 0.6299043
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5235238, 0.5241506
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5073462, 0.5093586
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7071905, 0.7051659
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6424103, 0.6397896
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5101957, 0.5079668
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6003122, 0.6050034
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6052346, 0.6036019
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4690199, 0.4701562
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7944071, 0.7904820

Time for backsubstitution: 23.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 752
type: DSZ, layer: 3, pos: 1795
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 1467
type: DSZ, layer: 3, pos: 1115
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1443

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2914

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2997111, upper bound: 0.3022570
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3013807, upper bound: 0.3008710
time: 5.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6296058, 0.6257551
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5219197, 0.5257554
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5082202, 0.5084844
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7053289, 0.7070270
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6395130, 0.6426880
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5091219, 0.5090408
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6050262, 0.6002905
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6012559, 0.6075809
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4693434, 0.4698327
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7924664, 0.7924228

Time for backsubstitution: 23.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2833
type: DSZ, layer: 3, pos: 1396
type: DSZ, layer: 3, pos: 1115
type: DSZ, layer: 3, pos: 1487
type: DSZ, layer: 3, pos: 1845
type: DSZ, layer: 3, pos: 1249
type: DSZ, layer: 3, pos: 1732
type: DSZ, layer: 3, pos: 1795
type: DSZ, layer: 3, pos: 752
type: DSZ, layer: 3, pos: 1698
type: DSZ, layer: 3, pos: 2242
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 2914
type: DSZ, layer: 3, pos: 163
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1467
type: DSZ, layer: 3, pos: 773
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 2123
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 758

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2833

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3013010, upper bound: 0.2967793
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.2959202, upper bound: 0.3021666
time: 3.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6267209, 0.6286395
1: -7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5257573, 0.5219176
2: -7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5084195, 0.5082850
3: -5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7086091, 0.7037468
4: -7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6408195, 0.6413817
5: -0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5089626, 0.5092001
6: -2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6003008, 0.6050165
7: -10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6062441, 0.6025922
8: 7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4698558, 0.4693202
9: -5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7929027, 0.7919862

Time for backsubstitution: 23.72 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.91 + 546.51 = 605.41 seconds
