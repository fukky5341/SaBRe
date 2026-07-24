## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.3532293525


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5333910, 2.5333900)
1: (-12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5630426, 2.5630426)
2: (-13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5471597, 2.5471601)
3: (-9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7141438, 2.7141433)
4: (-4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5921633, 1.5921636)
5: (-11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5612335, 2.5612335)
6: (-17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8651266, 2.8651257)
7: (-6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1917086, 2.1917083)
8: (-2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7734652, 1.7734652)
9: (2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2999578, 2.2999575)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.23 + 33.82 = 57.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -1.3600280, upper bound: 1.3600274

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6222
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 5747
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 6222

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3586187, upper bound: 1.3600261
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3600266, upper bound: 1.3586183
time: 4.23 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.57 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.57
Output dim: 9, lower bound: -1.3586187, upper bound: 1.3600261
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.57
Output dim: 9, lower bound: -1.3600266, upper bound: 1.3586183

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5278606, 2.5297484
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5603676, 2.5612783
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5310340, 2.5365400
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7065973, 2.7026911
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5871208, 1.5888433
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5478182, 2.5408754
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8627272, 2.8635459
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1857967, 2.1878104
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7680435, 1.7652435
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2966590, 2.2977839

Time for backsubstitution: 21.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 5747
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5816

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3581310, upper bound: 1.3600216
time: 5.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3586147, upper bound: 1.3595384
time: 5.03 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5297489, 2.5278602
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5612783, 2.5603685
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5365396, 2.5310335
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7026911, 2.7065973
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5888436, 1.5871205
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5408754, 2.5478182
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8635464, 2.8627276
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1878109, 2.1857965
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7652435, 1.7680435
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2977834, 2.2966590

Time for backsubstitution: 21.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 5747
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5816

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3595386, upper bound: 1.3586137
time: 5.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3600226, upper bound: 1.3581333
time: 5.59 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.88 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.88
Output dim: 9, lower bound: -1.3581310, upper bound: 1.3600216
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.88
Output dim: 9, lower bound: -1.3586147, upper bound: 1.3595384
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.88
Output dim: 9, lower bound: -1.3595386, upper bound: 1.3586137
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.88
Output dim: 9, lower bound: -1.3600226, upper bound: 1.3581333

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5132976, 2.5095148
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5531940, 2.5513277
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5242152, 2.5316286
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7060928, 2.7023263
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5776546, 1.5756881
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5399427, 2.5352001
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8470469, 2.8417635
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1855650, 2.1874914
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7648940, 1.7608685
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2917042, 2.2942183

Time for backsubstitution: 21.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5747
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5747

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3581160, upper bound: 1.3513992
time: 4.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3495098, upper bound: 1.3600074
time: 4.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5076261, 2.5151854
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5504179, 2.5541039
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5261226, 2.5297213
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7062330, 2.7021871
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5739653, 1.5793772
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5421419, 2.5329995
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8409452, 2.8478656
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1854782, 2.1875782
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7636685, 1.7620935
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2930937, 2.2928290

Time for backsubstitution: 21.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5747
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5747

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3585998, upper bound: 1.3509161
time: 4.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3499934, upper bound: 1.3595234
time: 12.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5151858, 2.5076265
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5541039, 2.5504179
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5297208, 2.5261226
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7021866, 2.7062325
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5793769, 1.5739655
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5330000, 2.5421419
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8478651, 2.8409448
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1875792, 2.1854775
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7620935, 1.7636688
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2928290, 2.2930937

Time for backsubstitution: 21.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5747
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5747

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3595236, upper bound: 1.3499923
time: 4.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3509162, upper bound: 1.3586021
time: 4.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5095143, 2.5132971
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5513277, 2.5531940
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5316291, 2.5242147
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7023268, 2.7060933
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5756881, 1.5776548
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5352001, 2.5399423
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8417635, 2.8470473
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1874924, 2.1855643
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7608685, 1.7648940
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2942181, 2.2917044

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5747
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5747

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3600076, upper bound: 1.3495100
time: 6.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3513997, upper bound: 1.3581153
time: 6.58 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 34.57 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.57
Output dim: 9, lower bound: -1.3581160, upper bound: 1.3513992
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.57
Output dim: 9, lower bound: -1.3495098, upper bound: 1.3600074
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.57
Output dim: 9, lower bound: -1.3585998, upper bound: 1.3509161
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.57
Output dim: 9, lower bound: -1.3499934, upper bound: 1.3595234
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.57
Output dim: 9, lower bound: -1.3595236, upper bound: 1.3499923
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.57
Output dim: 9, lower bound: -1.3509162, upper bound: 1.3586021
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.57
Output dim: 9, lower bound: -1.3600076, upper bound: 1.3495100
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.57
Output dim: 9, lower bound: -1.3513997, upper bound: 1.3581153

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5155606, 2.5129151
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5209804, 2.5231428
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5366054, 2.5472703
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7258835, 2.7198892
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5244551, 1.5291519
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5491915, 2.5456233
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8292718, 2.8262110
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1714077, 2.1694036
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7493610, 1.7431173
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2695541, 2.2689044

Time for backsubstitution: 21.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 833

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3578340, upper bound: 1.3502309
time: 5.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3483461, upper bound: 1.3502388
time: 5.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5166984, 2.5117774
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5250087, 2.5191150
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5398555, 2.5440192
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7236557, 2.7221174
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5311184, 1.5224886
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5503654, 2.5444498
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8314939, 2.8239894
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1674767, 2.1733344
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7471428, 1.7453353
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2663908, 2.2720678

Time for backsubstitution: 22.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 833

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3483477, upper bound: 1.3502369
time: 5.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3483402, upper bound: 1.3597247
time: 5.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5098891, 2.5185862
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5182052, 2.5259190
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5385127, 2.5453625
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7260237, 2.7197495
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5207658, 1.5328410
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5513906, 2.5434227
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8231702, 2.8323131
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1713209, 2.1694903
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7481356, 1.7443421
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2709432, 2.2675152

Time for backsubstitution: 22.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 833

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3583195, upper bound: 1.3497476
time: 6.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3488306, upper bound: 1.3497570
time: 7.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5110269, 2.5174484
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5222316, 2.5218911
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5417638, 2.5421119
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7237949, 2.7219777
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5274291, 1.5261776
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5525646, 2.5422487
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8253922, 2.8300910
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1673899, 2.1734211
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7459173, 1.7465601
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2677803, 2.2706785

Time for backsubstitution: 22.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 833

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3488322, upper bound: 1.3497527
time: 8.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3488247, upper bound: 1.3592396
time: 5.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5174489, 2.5110273
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5218902, 2.5222325
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5421119, 2.5417638
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7219772, 2.7237954
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5261774, 1.5274293
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5422487, 2.5525646
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8300900, 2.8253922
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1734209, 2.1673896
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7465601, 1.7459173
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2706785, 2.2677798

Time for backsubstitution: 22.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 833

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3592394, upper bound: 1.3488238
time: 5.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3497532, upper bound: 1.3488315
time: 5.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5185857, 2.5098891
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5259185, 2.5182047
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5453620, 2.5385132
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7197495, 2.7260237
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5328407, 1.5207660
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5434227, 2.5513911
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8323121, 2.8231707
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1694908, 2.1713204
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7443419, 1.7481356
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2675157, 2.2709429

Time for backsubstitution: 24.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 833

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3497548, upper bound: 1.3488330
time: 6.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3497473, upper bound: 1.3583193
time: 4.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5117774, 2.5166984
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5191150, 2.5250087
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5440192, 2.5398560
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7221174, 2.7236557
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5224886, 1.5311186
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5444498, 2.5503654
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8239884, 2.8314943
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1733341, 2.1674764
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7453356, 1.7471428
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2720680, 2.2663906

Time for backsubstitution: 22.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 833

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597255, upper bound: 1.3483398
time: 4.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3502378, upper bound: 1.3483479
time: 12.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5129151, 2.5155606
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5231433, 2.5209808
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5472693, 2.5366054
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7198887, 2.7258840
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5291519, 1.5244553
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5456228, 2.5491915
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8262105, 2.8292727
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1694040, 2.1714072
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7431173, 1.7493608
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2689047, 2.2695539

Time for backsubstitution: 23.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 833

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3502394, upper bound: 1.3483462
time: 6.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3502318, upper bound: 1.3578350
time: 7.15 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 37.74 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.74
Output dim: 9, lower bound: -1.3578340, upper bound: 1.3502309
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 37.74
Output dim: 9, lower bound: -1.3483461, upper bound: 1.3502388
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 37.74
Output dim: 9, lower bound: -1.3483477, upper bound: 1.3502369
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.74
Output dim: 9, lower bound: -1.3483402, upper bound: 1.3597247
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.74
Output dim: 9, lower bound: -1.3583195, upper bound: 1.3497476
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 37.74
Output dim: 9, lower bound: -1.3488306, upper bound: 1.3497570
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 37.74
Output dim: 9, lower bound: -1.3488322, upper bound: 1.3497527
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.74
Output dim: 9, lower bound: -1.3488247, upper bound: 1.3592396
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.74
Output dim: 9, lower bound: -1.3592394, upper bound: 1.3488238
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 37.74
Output dim: 9, lower bound: -1.3497532, upper bound: 1.3488315
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 37.74
Output dim: 9, lower bound: -1.3497548, upper bound: 1.3488330
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.74
Output dim: 9, lower bound: -1.3497473, upper bound: 1.3583193
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.74
Output dim: 9, lower bound: -1.3597255, upper bound: 1.3483398
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 37.74
Output dim: 9, lower bound: -1.3502378, upper bound: 1.3483479
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 37.74
Output dim: 9, lower bound: -1.3502394, upper bound: 1.3483462
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.74
Output dim: 9, lower bound: -1.3502318, upper bound: 1.3578350

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5177155, 2.5154190
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5141926, 2.5172029
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5414367, 2.5528831
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7311897, 2.7244573
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5124164, 1.5188756
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5507345, 2.5474148
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8244948, 2.8220291
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1775155, 2.1743279
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7460513, 1.7393348
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2635124, 2.2619998

Time for backsubstitution: 22.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 902

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3578285, upper bound: 1.3502278
time: 4.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3575358, upper bound: 1.3497424
time: 4.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5192013, 2.5139327
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5190697, 2.5123267
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5454698, 2.5488510
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7282238, 2.7274241
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5208421, 1.5104499
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5521564, 2.5459924
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8273129, 2.8192110
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1724010, 2.1794424
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7433600, 1.7420256
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2594860, 2.2660265

Time for backsubstitution: 22.74 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.05 + 546.30 = 603.35 seconds
