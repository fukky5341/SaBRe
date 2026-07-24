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
execution time: IAR + RelationalAnalysis = 23.03 + 33.49 = 56.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -1.3600280, upper bound: 1.3600274

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6222
type: DSZ, layer: 1, pos: 5747
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6222

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3586187, upper bound: 1.3600261
time: 4.17 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3600266, upper bound: 1.3586183
time: 4.25 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.43 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.43
Output dim: 9, lower bound: -1.3586187, upper bound: 1.3600261
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.43
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

Time for backsubstitution: 22.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5747
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5747

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3586037, upper bound: 1.3514033
time: 4.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3499974, upper bound: 1.3600112
time: 4.17 seconds

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

Time for backsubstitution: 21.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 5747
type: DSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 902

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3595402, upper bound: 1.3586142
time: 4.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3600227, upper bound: 1.3581322
time: 4.08 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.76 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.76
Output dim: 9, lower bound: -1.3586037, upper bound: 1.3514033
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.76
Output dim: 9, lower bound: -1.3499974, upper bound: 1.3600112
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.76
Output dim: 9, lower bound: -1.3595402, upper bound: 1.3586142
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.76
Output dim: 9, lower bound: -1.3600227, upper bound: 1.3581322

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5301247, 2.5331502
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5281544, 2.5330925
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5434241, 2.5521812
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7263870, 2.7202530
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5339217, 1.5423074
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5570679, 2.5512991
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8449540, 2.8479943
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1716394, 2.1697221
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7525115, 1.7474933
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2745085, 2.2724700

Time for backsubstitution: 22.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 902

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3581176, upper bound: 1.3513997
time: 4.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3585999, upper bound: 1.3509171
time: 4.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5312614, 2.5320125
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5321827, 2.5290647
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5466743, 2.5489306
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7241592, 2.7224813
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5405850, 1.5356441
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5582418, 2.5501251
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8471761, 2.8457727
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1677084, 2.1736529
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7502937, 1.7497115
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2713451, 2.2756333

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5816
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 833

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5816

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3495098, upper bound: 1.3600074
time: 4.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3499934, upper bound: 1.3595234
time: 12.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5282087, 2.5248165
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5607166, 2.5592742
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5336947, 2.5295997
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7026863, 2.7065892
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5880938, 1.5856352
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5400639, 2.5474029
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8616562, 2.8589931
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1875801, 2.1853454
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7649012, 1.7673705
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2964621, 2.2959900

Time for backsubstitution: 21.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5747
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5747

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3595253, upper bound: 1.3499928
time: 4.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3509178, upper bound: 1.3585994
time: 4.66 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5267038, 2.5263209
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5601845, 2.5598063
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5351062, 2.5281882
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7026825, 2.7065935
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5873575, 1.5863712
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5404606, 2.5470061
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8598108, 2.8608379
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1873598, 2.1855659
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7645702, 1.7677011
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2971148, 2.2953374

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5747
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5747

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3600078, upper bound: 1.3495113
time: 4.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3513999, upper bound: 1.3581169
time: 5.11 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.96 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.96
Output dim: 9, lower bound: -1.3581176, upper bound: 1.3513997
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.96
Output dim: 9, lower bound: -1.3585999, upper bound: 1.3509171
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.96
Output dim: 9, lower bound: -1.3495098, upper bound: 1.3600074
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.96
Output dim: 9, lower bound: -1.3499934, upper bound: 1.3595234
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.96
Output dim: 9, lower bound: -1.3595253, upper bound: 1.3499928
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.96
Output dim: 9, lower bound: -1.3509178, upper bound: 1.3585994
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.96
Output dim: 9, lower bound: -1.3600078, upper bound: 1.3495113
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.96
Output dim: 9, lower bound: -1.3513999, upper bound: 1.3581169

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5285840, 2.5301061
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5275931, 2.5319991
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5405784, 2.5507469
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7263832, 2.7202449
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5331721, 1.5408220
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5562563, 2.5508833
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8430638, 2.8442593
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1714087, 2.1692710
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7521687, 1.7468200
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2731872, 2.2718010

Time for backsubstitution: 22.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 833

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3578356, upper bound: 1.3502344
time: 4.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3483477, upper bound: 1.3502394
time: 6.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5270801, 2.5316105
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5270610, 2.5325308
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5419898, 2.5493355
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7263794, 2.7202492
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5324359, 1.5415583
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5566530, 2.5504870
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8412185, 2.8461041
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1711884, 2.1694915
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7518382, 1.7471507
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2738400, 2.2711482

Time for backsubstitution: 22.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 833

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3583196, upper bound: 1.3497504
time: 4.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3488307, upper bound: 1.3497588
time: 4.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 22.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 833

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 902

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3495044, upper bound: 1.3600040
time: 7.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3492141, upper bound: 1.3595190
time: 4.89 seconds

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

Time for backsubstitution: 22.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 902

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 833

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3488322, upper bound: 1.3497525
time: 8.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3488247, upper bound: 1.3592396
time: 5.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5304723, 2.5282183
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5285029, 2.5310888
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5460849, 2.5452409
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7224770, 2.7241511
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5348945, 1.5390992
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5493135, 2.5578260
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8438821, 2.8434410
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1734228, 2.1672571
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7493687, 1.7496200
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2743120, 2.2706761

Time for backsubstitution: 22.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 833

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3592410, upper bound: 1.3488245
time: 4.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3497548, upper bound: 1.3488324
time: 4.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5316110, 2.5270801
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5325313, 2.5270610
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5493350, 2.5419903
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7202492, 2.7263794
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5415578, 1.5324359
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5504875, 2.5566525
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8461041, 2.8412189
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1694918, 2.1711879
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7471509, 1.7518382
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2711487, 2.2738395

Time for backsubstitution: 21.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 833

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3497564, upper bound: 1.3488307
time: 4.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3497488, upper bound: 1.3583198
time: 4.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5289683, 2.5297222
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5279708, 2.5316205
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5474963, 2.5438290
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7224731, 2.7241554
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5341582, 1.5398355
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5497103, 2.5574298
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8420377, 2.8452859
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1732025, 2.1674776
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7490382, 1.7499506
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2749643, 2.2700236

Time for backsubstitution: 22.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 833

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3597257, upper bound: 1.3483442
time: 4.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3502379, upper bound: 1.3483493
time: 4.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5301061, 2.5285845
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5319991, 2.5275927
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5507464, 2.5405788
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7202454, 2.7263832
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5408216, 1.5331721
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5508833, 2.5562563
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8442588, 2.8430643
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1692715, 2.1714084
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7468200, 1.7521687
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2718010, 2.2731869

Time for backsubstitution: 23.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 833

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.3502396, upper bound: 1.3483470
time: 4.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3502320, upper bound: 1.3578347
time: 4.45 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.81 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.81
Output dim: 9, lower bound: -1.3578356, upper bound: 1.3502344
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.81
Output dim: 9, lower bound: -1.3483477, upper bound: 1.3502394
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.81
Output dim: 9, lower bound: -1.3583196, upper bound: 1.3497504
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.81
Output dim: 9, lower bound: -1.3488307, upper bound: 1.3497588
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.81
Output dim: 9, lower bound: -1.3495044, upper bound: 1.3600040
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.81
Output dim: 9, lower bound: -1.3492141, upper bound: 1.3595190
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.81
Output dim: 9, lower bound: -1.3488322, upper bound: 1.3497525
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.81
Output dim: 9, lower bound: -1.3488247, upper bound: 1.3592396
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.81
Output dim: 9, lower bound: -1.3592410, upper bound: 1.3488245
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.81
Output dim: 9, lower bound: -1.3497548, upper bound: 1.3488324
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.81
Output dim: 9, lower bound: -1.3497564, upper bound: 1.3488307
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.81
Output dim: 9, lower bound: -1.3497488, upper bound: 1.3583198
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.81
Output dim: 9, lower bound: -1.3597257, upper bound: 1.3483442
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.81
Output dim: 9, lower bound: -1.3502379, upper bound: 1.3483493
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.81
Output dim: 9, lower bound: -1.3502396, upper bound: 1.3483470
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.81
Output dim: 9, lower bound: -1.3502320, upper bound: 1.3578347

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5307388, 2.5326099
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5208039, 2.5260592
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5454106, 2.5563612
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7316895, 2.7248130
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5211334, 1.5305459
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5577965, 2.5526733
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8382845, 2.8400764
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1775165, 2.1741951
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7488580, 1.7430367
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2671459, 2.2648962

Time for backsubstitution: 23.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5816

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3578285, upper bound: 1.3502278
time: 4.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3578298, upper bound: 1.3495571
time: 5.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5292358, 2.5341144
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5202718, 2.5265903
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5468221, 2.5549498
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7316856, 2.7248168
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5203972, 1.5312822
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5581932, 2.5522771
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8364401, 2.8419213
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1772957, 2.1744156
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7485275, 1.7433672
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2677982, 2.2642434

Time for backsubstitution: 23.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5816

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3575358, upper bound: 1.3497424
time: 4.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.3583157, upper bound: 1.3497433
time: 4.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.3000660, -10.2871666, -14.3000660, -10.2871666, -2.5151596, 2.5087342
1: -12.4945765, -8.9361649, -12.4945765, -8.9361649, -2.5244546, 2.5180216
2: -13.4097614, -10.1796112, -13.4097614, -10.1796112, -2.5370107, 2.5425858
3: -9.8902378, -6.9025407, -9.8902378, -6.9025407, -2.7236519, 2.7221093
4: -4.5608406, -2.3997998, -4.5608406, -2.3997998, -1.5303693, 1.5210032
5: -11.0733929, -7.3661022, -11.0733929, -7.3661022, -2.5495558, 2.5440412
6: -17.5802193, -13.6031485, -17.5802193, -13.6031485, -2.8296046, 2.8202548
7: -6.4332128, -3.5954399, -6.4332128, -3.5954399, -2.1672478, 2.1728823
8: -2.0399046, 0.1837788, -2.0399046, 0.1837788, -1.7468019, 1.7446620
9: 2.4171548, 5.1602297, 2.4171548, 5.1602297, -2.2650695, 2.2713995

Time for backsubstitution: 23.98 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.52 + 549.76 = 606.28 seconds
