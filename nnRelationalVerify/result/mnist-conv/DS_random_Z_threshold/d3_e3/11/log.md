## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.441476739


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5637841, 1.5637841)
1: (-11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1966600, 1.1966598)
2: (-7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9311540, 0.9311540)
3: (-7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.4207768, 1.4207768)
4: (-3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.4014397, 1.4014397)
5: (-5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2628298, 1.2628298)
6: (-16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2654819, 1.2654817)
7: (-4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3419938, 1.3419938)
8: (-4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9227247, 0.9227247)
9: (4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7659516, 0.7659515)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.06 + 34.56 = 56.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.4459361, upper bound: 0.4459358

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4670
type: DSZ, layer: 1, pos: 4628
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 5799
type: DSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4670

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459349, upper bound: 0.4455780
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4455786, upper bound: 0.4459348
time: 3.67 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.34 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.34
Output dim: 9, lower bound: -0.4459349, upper bound: 0.4455780
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.34
Output dim: 9, lower bound: -0.4455786, upper bound: 0.4459348

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5548921, 1.5581884
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1947393, 1.1954498
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9264755, 0.9282125
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.4153485, 1.4173665
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.4011192, 1.4009297
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2579341, 1.2597528
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2589512, 1.2613771
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3419170, 1.3419394
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9212823, 0.9204369
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7649829, 0.7644099

Time for backsubstitution: 20.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 6166
type: DSZ, layer: 1, pos: 4628
type: DSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4449042, upper bound: 0.4455772
time: 3.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459341, upper bound: 0.4445501
time: 3.69 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5581884, 1.5548918
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1954498, 1.1947398
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9282124, 0.9264755
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.4173670, 1.4153485
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.4009295, 1.4011192
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2597528, 1.2579341
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2613773, 1.2589514
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3419394, 1.3419173
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9204369, 0.9212824
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7644098, 0.7649828

Time for backsubstitution: 20.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5799
type: DSZ, layer: 1, pos: 6166
type: DSZ, layer: 1, pos: 4628
type: DSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5799

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4444952, upper bound: 0.4459366
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4455782, upper bound: 0.4448539
time: 2.96 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 26.47 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 26.47
Output dim: 9, lower bound: -0.4449042, upper bound: 0.4455772
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 26.47
Output dim: 9, lower bound: -0.4459341, upper bound: 0.4445501
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 26.47
Output dim: 9, lower bound: -0.4444952, upper bound: 0.4459366
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 26.47
Output dim: 9, lower bound: -0.4455782, upper bound: 0.4448539

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5525427, 1.5553820
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1951632, 1.1960056
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9268374, 0.9286873
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.4105430, 1.4133615
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3981261, 1.3973362
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2523365, 1.2530344
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2535009, 1.2548518
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3332276, 1.3347001
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9159317, 0.9159782
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7623887, 0.7622459

Time for backsubstitution: 20.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6166
type: DSZ, layer: 1, pos: 4628
type: DSZ, layer: 1, pos: 5799

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4449033, upper bound: 0.4450684
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4443942, upper bound: 0.4455768
time: 3.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5520854, 1.5558391
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1952958, 1.1958730
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9269505, 0.9285744
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.4113431, 1.4125609
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3975263, 1.3979363
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2512159, 1.2541547
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2524257, 1.2559266
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3346782, 1.3332500
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9168236, 0.9150863
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7628191, 0.7618158

Time for backsubstitution: 20.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5799
type: DSZ, layer: 1, pos: 6166
type: DSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5799

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4448491, upper bound: 0.4444995
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459337, upper bound: 0.4444936
time: 3.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5537677, 1.5495977
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1990337, 1.1990521
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9286869, 0.9270461
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.4009371, 1.4016550
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3899674, 1.3879662
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2598896, 1.2547207
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2465038, 1.2411115
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3133216, 1.3180699
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9095645, 0.9132084
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7654097, 0.7676179

Time for backsubstitution: 20.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 4628
type: DSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4444937, upper bound: 0.4459358
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4444923, upper bound: 0.4448548
time: 3.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5528941, 1.5504708
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1997623, 1.1983237
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9287832, 0.9269497
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.4036732, 1.3989186
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3877769, 1.3901572
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2565393, 1.2580709
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2435374, 1.2440784
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3180919, 1.3132997
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9123628, 0.9104098
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7670448, 0.7659827

Time for backsubstitution: 20.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4628
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4628

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4444262, upper bound: 0.4448516
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4455759, upper bound: 0.4437019
time: 3.02 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 26.43 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.43
Output dim: 9, lower bound: -0.4449033, upper bound: 0.4450684
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.43
Output dim: 9, lower bound: -0.4443942, upper bound: 0.4455768
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.43
Output dim: 9, lower bound: -0.4448491, upper bound: 0.4444995
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.43
Output dim: 9, lower bound: -0.4459337, upper bound: 0.4444936
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.43
Output dim: 9, lower bound: -0.4444937, upper bound: 0.4459358
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.43
Output dim: 9, lower bound: -0.4444923, upper bound: 0.4448548
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 26.43
Output dim: 9, lower bound: -0.4444262, upper bound: 0.4448516
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.43
Output dim: 9, lower bound: -0.4455759, upper bound: 0.4437019

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5162106, 1.5090778
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1854177, 1.1822238
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8861713, 0.8949332
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3595991, 1.3521991
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.4038215, 1.4054890
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2173486, 1.2110064
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2458568, 1.2450719
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3296337, 1.3317058
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9162788, 0.9168900
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7625382, 0.7624295

Time for backsubstitution: 20.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5799
type: DSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5799

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4448495, upper bound: 0.4450685
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4448540, upper bound: 0.4439835
time: 3.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5062385, 1.5190501
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1813807, 1.1862605
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8930836, 0.8880213
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3493805, 1.3624177
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.4062786, 1.4030321
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2103086, 1.2180464
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2437210, 1.2472079
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3302336, 1.3311064
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9168434, 0.9163254
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7625723, 0.7623954

Time for backsubstitution: 20.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5799
type: DSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5799

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4443396, upper bound: 0.4455784
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4443441, upper bound: 0.4444924
time: 3.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5491672, 1.5520477
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1988802, 1.2001858
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9274251, 0.9291455
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3949122, 1.3988662
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3865652, 1.3847842
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2616472, 1.2612360
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2375538, 1.2380874
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3060641, 1.3094068
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9141657, 0.9152268
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7678018, 0.7684337

Time for backsubstitution: 20.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6166
type: DSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4448482, upper bound: 0.4439893
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4443383, upper bound: 0.4444971
time: 4.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5482941, 1.5529296
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1996083, 1.1994576
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9275215, 0.9290490
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3976502, 1.3961298
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3843737, 1.3869750
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2582970, 1.2645860
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2345870, 1.2410631
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3108349, 1.3046365
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9169643, 0.9124284
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7694380, 0.7667985

Time for backsubstitution: 20.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4628
type: DSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4628

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4447817, upper bound: 0.4444934
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459315, upper bound: 0.4433437
time: 3.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5529299, 1.5482941
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1994576, 1.1996083
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9290490, 0.9275215
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3961301, 1.3976502
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3869753, 1.3843739
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2645855, 1.2582970
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2410634, 1.2345870
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3046365, 1.3108349
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9124281, 0.9169643
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7667987, 0.7694380

Time for backsubstitution: 20.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4628
type: DSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4628

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4433417, upper bound: 0.4459329
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4444914, upper bound: 0.4447838
time: 3.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5524635, 1.5487509
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1995902, 1.1994758
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9291620, 0.9274085
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3969302, 1.3968477
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3863754, 1.3849738
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2634659, 1.2594175
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2399795, 1.2356617
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3060865, 1.3093846
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9133203, 0.9160722
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7672288, 0.7690067

Time for backsubstitution: 20.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6166
type: DSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6166

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4444914, upper bound: 0.4443461
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4439815, upper bound: 0.4448559
time: 3.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5450077, 1.5389090
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1979852, 1.1961949
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9331841, 0.9306527
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3887672, 1.3864851
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3886499, 1.3913140
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2589602, 1.2601061
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2438278, 1.2420423
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3148317, 1.3093951
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9053018, 0.9019305
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7603722, 0.7605276

Time for backsubstitution: 20.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6166
type: DSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4444253, upper bound: 0.4443408
time: 3.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4439154, upper bound: 0.4448502
time: 2.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5413322, 1.5425844
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1976333, 1.1965468
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9324861, 0.9313509
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3912396, 1.3840127
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3889332, 1.3910306
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2585745, 1.2604921
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2415009, 1.2443688
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3141880, 1.3100393
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9038832, 0.9033490
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7615898, 0.7593101

Time for backsubstitution: 20.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 6166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 885

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4444959, upper bound: 0.4436966
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4455751, upper bound: 0.4437001
time: 2.98 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 27.90 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 9, lower bound: -0.4448495, upper bound: 0.4450685
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 9, lower bound: -0.4448540, upper bound: 0.4439835
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 9, lower bound: -0.4443396, upper bound: 0.4455784
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 9, lower bound: -0.4443441, upper bound: 0.4444924
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 9, lower bound: -0.4448482, upper bound: 0.4439893
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 9, lower bound: -0.4443383, upper bound: 0.4444971
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 9, lower bound: -0.4447817, upper bound: 0.4444934
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 9, lower bound: -0.4459315, upper bound: 0.4433437
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 9, lower bound: -0.4433417, upper bound: 0.4459329
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 9, lower bound: -0.4444914, upper bound: 0.4447838
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 9, lower bound: -0.4444914, upper bound: 0.4443461
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 9, lower bound: -0.4439815, upper bound: 0.4448559
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 9, lower bound: -0.4444253, upper bound: 0.4443408
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 9, lower bound: -0.4439154, upper bound: 0.4448502
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 9, lower bound: -0.4444959, upper bound: 0.4436966
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 27.90
Output dim: 9, lower bound: -0.4455751, upper bound: 0.4437001

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5133014, 1.5052862
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1890016, 1.1865358
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8866453, 0.8955035
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3431683, 1.3385065
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3928599, 1.3923364
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2277799, 1.2180877
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2309935, 1.2272327
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3010206, 1.3078630
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9136214, 0.9170310
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7675209, 0.7690485

Time for backsubstitution: 20.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4628

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4436975, upper bound: 0.4450663
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4448472, upper bound: 0.4439165
time: 3.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5124192, 1.5061593
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1897302, 1.1858077
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8867416, 0.8954071
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3459044, 1.3357685
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3906693, 1.3945272
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2244296, 1.2214379
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2280176, 1.2301996
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3057909, 1.3030922
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9164200, 0.9142325
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7691562, 0.7674122

Time for backsubstitution: 20.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4628

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4437020, upper bound: 0.4439812
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4448517, upper bound: 0.4428315
time: 3.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5033288, 1.5152588
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1849651, 1.1905727
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8935571, 0.8885915
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3329496, 1.3487251
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3953166, 1.3898795
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2207398, 1.2251277
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2288578, 1.2293687
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3016195, 1.3072631
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9141860, 0.9164665
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7675552, 0.7690144

Time for backsubstitution: 20.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4628

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4431876, upper bound: 0.4455761
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4443374, upper bound: 0.4444264
time: 3.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5024467, 1.5161319
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1856933, 1.1898444
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8936534, 0.8884952
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3356857, 1.3459871
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3931260, 1.3920705
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2173896, 1.2284777
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2258818, 1.2323356
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3063898, 1.3024924
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9169846, 0.9136679
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7691903, 0.7673781

Time for backsubstitution: 20.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4628

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4431921, upper bound: 0.4444911
time: 2.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4443418, upper bound: 0.4433408
time: 3.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5128355, 1.5057435
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1891341, 1.1864035
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8867581, 0.8953905
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3439684, 1.3377044
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3922601, 1.3929362
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2266593, 1.2192080
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2299097, 1.2283075
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3024702, 1.3064125
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9145136, 0.9161390
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7679513, 0.7686172

Time for backsubstitution: 20.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4628

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4436962, upper bound: 0.4439847
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4448459, upper bound: 0.4428369
time: 3.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5028629, 1.5157158
1: -11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1850977, 1.1904402
2: -7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.8936701, 0.8884785
3: -7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.3337498, 1.3479230
4: -3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.3947167, 1.3904796
5: -5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2196193, 1.2262480
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2277739, 1.2304435
7: -4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3030701, 1.3058128
8: -4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9150782, 0.9155746
9: 4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7679853, 0.7685831

Time for backsubstitution: 20.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4628

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4628

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4431863, upper bound: 0.4444946
time: 4.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4443360, upper bound: 0.4433472
time: 3.02 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 28.42 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.42
Output dim: 9, lower bound: -0.4436975, upper bound: 0.4450663
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.42
Output dim: 9, lower bound: -0.4448472, upper bound: 0.4439165
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.42
Output dim: 9, lower bound: -0.4437020, upper bound: 0.4439812
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.42
Output dim: 9, lower bound: -0.4448517, upper bound: 0.4428315
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.42
Output dim: 9, lower bound: -0.4431876, upper bound: 0.4455761
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.42
Output dim: 9, lower bound: -0.4443374, upper bound: 0.4444264
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.42
Output dim: 9, lower bound: -0.4431921, upper bound: 0.4444911
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.42
Output dim: 9, lower bound: -0.4443418, upper bound: 0.4433408
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.42
Output dim: 9, lower bound: -0.4436962, upper bound: 0.4439847
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.42
Output dim: 9, lower bound: -0.4448459, upper bound: 0.4428369
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.42
Output dim: 9, lower bound: -0.4431863, upper bound: 0.4444946
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.42
Output dim: 9, lower bound: -0.4443360, upper bound: 0.4433472
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.42
Output dim: 9, lower bound: -0.4447817, upper bound: 0.4444934
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.42
Output dim: 9, lower bound: -0.4459315, upper bound: 0.4433437
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.42
Output dim: 9, lower bound: -0.4433417, upper bound: 0.4459329
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.42
Output dim: 9, lower bound: -0.4444914, upper bound: 0.4447838
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.42
Output dim: 9, lower bound: -0.4444914, upper bound: 0.4443461
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.42
Output dim: 9, lower bound: -0.4439815, upper bound: 0.4448559
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.42
Output dim: 9, lower bound: -0.4444253, upper bound: 0.4443408
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.42
Output dim: 9, lower bound: -0.4439154, upper bound: 0.4448502
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.42
Output dim: 9, lower bound: -0.4444959, upper bound: 0.4436966
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.42
Output dim: 9, lower bound: -0.4455751, upper bound: 0.4437001

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.62 + 549.95 = 606.58 seconds
