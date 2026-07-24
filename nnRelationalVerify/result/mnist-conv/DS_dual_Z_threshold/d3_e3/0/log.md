## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.635617521


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5792060, 1.5792060)
1: (-6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4270267, 1.4270267)
2: (-8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3698587, 1.3698590)
3: (-9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.3052292, 1.3052292)
4: (-4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5216794, 1.5216794)
5: (-5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4439821, 1.4439824)
6: (-13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0481853, 2.0481853)
7: (3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1529670, 1.1529672)
8: (-4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.5045223, 1.5045223)
9: (-1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5415206, 1.5415206)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.15 + 38.57 = 61.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.6420365, upper bound: 0.6420360

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6235
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6235

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420352, upper bound: 0.6420342
time: 4.18 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420352, upper bound: 0.6420343
time: 4.18 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.63 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.63
Output dim: 7, lower bound: -0.6420352, upper bound: 0.6420342
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.63
Output dim: 7, lower bound: -0.6420352, upper bound: 0.6420343

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5752206, 1.5744357
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4232159, 1.4224544
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3702302, 1.3708036
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.3048131, 1.3048820
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5215235, 1.5216398
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4435692, 1.4430461
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0196567, 2.0244131
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1533861, 1.1536486
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.5010552, 1.4977660
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5412588, 1.5414495

Time for backsubstitution: 21.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 6208

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405160, upper bound: 0.6420306
time: 4.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420314, upper bound: 0.6405156
time: 4.03 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5744357, 1.5752206
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4224544, 1.4232159
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3708034, 1.3702304
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.3048823, 1.3048129
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5216403, 1.5215235
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4430461, 1.4435692
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0244136, 2.0196571
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1536489, 1.1533864
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4977660, 1.5010552
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5414495, 1.5412586

Time for backsubstitution: 21.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6208
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 6208

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405160, upper bound: 0.6420305
time: 4.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420314, upper bound: 0.6405156
time: 4.08 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.41 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.41
Output dim: 7, lower bound: -0.6405160, upper bound: 0.6420306
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.41
Output dim: 7, lower bound: -0.6420314, upper bound: 0.6405156
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.41
Output dim: 7, lower bound: -0.6405160, upper bound: 0.6420305
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.41
Output dim: 7, lower bound: -0.6420314, upper bound: 0.6405156

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5476227, 1.5588193
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4080091, 1.3955655
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3668356, 1.3648126
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.3039725, 1.3034003
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5096827, 1.5007048
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4223223, 1.4310291
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0153465, 2.0168123
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1506395, 1.1520901
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4985790, 1.4934082
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5380659, 1.5396397

Time for backsubstitution: 22.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 5857

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382819, upper bound: 0.6420296
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405145, upper bound: 0.6397958
time: 3.82 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5596037, 1.5468383
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.3963270, 1.4072473
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3642387, 1.3674095
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.3033311, 1.3040416
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5005879, 1.5097995
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4315529, 1.4217985
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0120564, 2.0201030
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1518278, 1.1509016
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4966974, 1.4952898
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5394487, 1.5382569

Time for backsubstitution: 22.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 5857

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6397970, upper bound: 0.6405160
time: 3.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420298, upper bound: 0.6382834
time: 5.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5468383, 1.5596037
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4072471, 1.3963270
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3674097, 1.3642390
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.3040416, 1.3033314
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5097995, 1.5005879
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4217987, 1.4315526
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0201035, 2.0120564
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1509018, 1.1518278
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4952898, 1.4966974
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5382566, 1.5394487

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 5857

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382819, upper bound: 0.6420296
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405145, upper bound: 0.6397958
time: 3.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5588193, 1.5476227
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.3955655, 1.4080088
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3648129, 1.3668358
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.3034003, 1.3039725
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5007048, 1.5096827
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4310293, 1.4223220
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0168133, 2.0153465
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1520901, 1.1506391
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4934082, 1.4985790
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5396395, 1.5380659

Time for backsubstitution: 22.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5857
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 5857

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6397970, upper bound: 0.6405159
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420298, upper bound: 0.6382834
time: 6.59 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 32.84 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.84
Output dim: 7, lower bound: -0.6382819, upper bound: 0.6420296
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.84
Output dim: 7, lower bound: -0.6405145, upper bound: 0.6397958
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.84
Output dim: 7, lower bound: -0.6397970, upper bound: 0.6405160
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.84
Output dim: 7, lower bound: -0.6420298, upper bound: 0.6382834
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.84
Output dim: 7, lower bound: -0.6382819, upper bound: 0.6420296
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.84
Output dim: 7, lower bound: -0.6405145, upper bound: 0.6397958
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.84
Output dim: 7, lower bound: -0.6397970, upper bound: 0.6405159
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.84
Output dim: 7, lower bound: -0.6420298, upper bound: 0.6382834

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5458879, 1.5505090
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4080052, 1.3955491
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3653231, 1.3575337
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.3022165, 1.2949450
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5072560, 1.5002074
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4205694, 1.4225948
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0076675, 2.0152240
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1484375, 1.1516328
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4971595, 1.4865561
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5332217, 1.5386348

Time for backsubstitution: 21.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6361796, upper bound: 0.6420266
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382788, upper bound: 0.6399275
time: 4.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5393124, 1.5570841
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4079924, 1.3955617
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3595572, 1.3632996
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2955170, 1.3016441
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5091863, 1.4982772
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4138870, 1.4292772
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0137577, 2.0091338
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1501818, 1.1498885
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4917274, 1.4919853
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5370612, 1.5347953

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 6181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6384128, upper bound: 0.6397936
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405114, upper bound: 0.6376936
time: 3.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5578690, 1.5385280
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.3963232, 1.4072309
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3627262, 1.3601305
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.3015752, 1.2955863
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4981608, 1.5093026
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4298000, 1.4133642
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0043774, 2.0185142
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1496258, 1.1504440
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4952779, 1.4884381
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5346045, 1.5372519

Time for backsubstitution: 22.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 6181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6376948, upper bound: 0.6405115
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6397940, upper bound: 0.6384144
time: 3.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5512934, 1.5451031
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.3963103, 1.4072437
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3569603, 1.3658960
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2948761, 1.3022854
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5000911, 1.5073724
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4231176, 1.4200466
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0104675, 2.0124245
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1513700, 1.1486998
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4898453, 1.4938669
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5384440, 1.5334125

Time for backsubstitution: 22.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 6181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6399279, upper bound: 0.6382777
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420268, upper bound: 0.6361812
time: 4.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5451031, 1.5512938
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4072437, 1.3963106
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3658962, 1.3569605
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.3022852, 1.2948761
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5073724, 1.5000911
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4200468, 1.4231179
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0124245, 2.0104671
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1486998, 1.1513700
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4938669, 1.4898453
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5334125, 1.5384440

Time for backsubstitution: 22.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 6181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6361796, upper bound: 0.6420265
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382788, upper bound: 0.6399271
time: 3.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5385280, 1.5578690
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4072309, 1.3963232
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3601303, 1.3627264
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2955866, 1.3015749
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5093026, 1.4981604
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4133644, 1.4298003
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0185146, 2.0043778
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1504440, 1.1496263
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4884381, 1.4952779
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5372519, 1.5346043

Time for backsubstitution: 22.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 6181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6384128, upper bound: 0.6397928
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6405114, upper bound: 0.6376936
time: 3.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5570841, 1.5393128
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.3955617, 1.4079924
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3632994, 1.3595569
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.3016443, 1.2955172
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.4982772, 1.5091863
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4292774, 1.4138873
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0091343, 2.0137572
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1498890, 1.1501815
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4919853, 1.4917274
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5347953, 1.5370612

Time for backsubstitution: 22.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 6181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6376948, upper bound: 0.6405116
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6397940, upper bound: 0.6384144
time: 3.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5505090, 1.5458879
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.3955493, 1.4080052
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3575335, 1.3653233
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2949452, 1.3022163
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5002079, 1.5072556
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.4225950, 1.4205697
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0152245, 2.0076680
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1516323, 1.1484375
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4865561, 1.4971595
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5386348, 1.5332215

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6181
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6399279, upper bound: 0.6382777
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6420268, upper bound: 0.6361812
time: 4.18 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.40 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 7, lower bound: -0.6361796, upper bound: 0.6420266
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 7, lower bound: -0.6382788, upper bound: 0.6399275
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 7, lower bound: -0.6384128, upper bound: 0.6397936
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 7, lower bound: -0.6405114, upper bound: 0.6376936
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 7, lower bound: -0.6376948, upper bound: 0.6405115
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 7, lower bound: -0.6397940, upper bound: 0.6384144
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 7, lower bound: -0.6399279, upper bound: 0.6382777
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 7, lower bound: -0.6420268, upper bound: 0.6361812
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 7, lower bound: -0.6361796, upper bound: 0.6420265
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 7, lower bound: -0.6382788, upper bound: 0.6399271
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 7, lower bound: -0.6384128, upper bound: 0.6397928
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 7, lower bound: -0.6405114, upper bound: 0.6376936
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 7, lower bound: -0.6376948, upper bound: 0.6405116
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 7, lower bound: -0.6397940, upper bound: 0.6384144
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 7, lower bound: -0.6399279, upper bound: 0.6382777
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.40
Output dim: 7, lower bound: -0.6420268, upper bound: 0.6361812

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5404358, 1.5459652
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4095616, 1.3962283
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3586249, 1.3524661
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2887154, 1.2836878
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5017505, 1.4971981
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.3873730, 1.3956072
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0203547, 2.0256457
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1485057, 1.1516330
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4813166, 1.4663396
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5304842, 1.5353544

Time for backsubstitution: 22.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 4627

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6356112, upper bound: 0.6420272
time: 3.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6361789, upper bound: 0.6414591
time: 3.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5413442, 1.5450568
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4086843, 1.3971057
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3602552, 1.3508358
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2909589, 1.2814438
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5042462, 1.4947023
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.3935804, 1.3893974
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0180907, 2.0279102
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1484380, 1.1517007
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4769425, 1.4707131
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5299411, 1.5358973

Time for backsubstitution: 22.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 4627

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6377091, upper bound: 0.6399265
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6382780, upper bound: 0.6393612
time: 3.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5338602, 1.5525403
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4095492, 1.3962412
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3528585, 1.3582315
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2820158, 1.2903869
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5036807, 1.4952679
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.3806906, 1.4022896
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0264440, 2.0195556
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1502500, 1.1498890
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4758844, 1.4717684
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5343237, 1.5315149

Time for backsubstitution: 22.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4627
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 4627

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6378444, upper bound: 0.6397943
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.6384121, upper bound: 0.6392243
time: 3.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8265705, -6.6989441, -8.8265705, -6.6989441, -1.5347686, 1.5516319
1: -6.3877430, -4.6722221, -6.3877430, -4.6722221, -1.4086719, 1.3971183
2: -8.4230862, -6.8273726, -8.4230862, -6.8273726, -1.3544893, 1.3566012
3: -9.7901926, -7.9756384, -9.7901926, -7.9756384, -1.2842598, 1.2881429
4: -4.6745596, -3.0056338, -4.6745596, -3.0056338, -1.5061769, 1.4927721
5: -5.0677209, -3.3530407, -5.0677209, -3.3530407, -1.3868980, 1.3960798
6: -13.3388367, -11.1216059, -13.3388367, -11.1216059, -2.0241790, 2.0218205
7: 3.5913880, 4.8264775, 3.5913880, 4.8264775, -1.1501822, 1.1499565
8: -4.0817289, -2.0028186, -4.0817289, -2.0028186, -1.4715104, 1.4761424
9: -1.9310472, -0.2524234, -1.9310472, -0.2524234, -1.5337811, 1.5320578

Time for backsubstitution: 22.29 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 61.72 + 547.51 = 609.23 seconds
