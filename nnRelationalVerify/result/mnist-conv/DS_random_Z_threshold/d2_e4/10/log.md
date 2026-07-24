## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.301048902


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8750806, 0.8750806)
1: (2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7192923, 0.7192924)
2: (-5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5813267, 0.5813267)
3: (-9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5080827, 0.5080827)
4: (-4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5576992, 0.5576992)
5: (-8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5290227, 0.5290227)
6: (-5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.9008565, 0.9008565)
7: (-3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7421594, 0.7421596)
8: (-3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4909875, 0.4909875)
9: (-10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8214734, 0.8214734)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.27 + 32.66 = 56.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.3040898, upper bound: 0.3040897

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5830
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 916

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5830

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3030403, upper bound: 0.3040872
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3040875, upper bound: 0.3030402
time: 2.68 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.37 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.37
Output dim: 1, lower bound: -0.3030403, upper bound: 0.3040872
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.37
Output dim: 1, lower bound: -0.3040875, upper bound: 0.3030402

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8728371, 0.8719065
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7151464, 0.7163627
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5734756, 0.5757806
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5074540, 0.5071943
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5519458, 0.5536340
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5274101, 0.5267409
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8955486, 0.8933468
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7364168, 0.7381032
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4860455, 0.4839951
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8187993, 0.8176906

Time for backsubstitution: 22.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 916

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3029732, upper bound: 0.3037898
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3029735, upper bound: 0.3029740
time: 2.64 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8719068, 0.8728368
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7163626, 0.7151465
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5757806, 0.5734756
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5071943, 0.5074540
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5536339, 0.5519458
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5267411, 0.5274104
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8933468, 0.8955486
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7381029, 0.7364168
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4839951, 0.4860455
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8176911, 0.8187997

Time for backsubstitution: 23.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 916

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3029741, upper bound: 0.3029734
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3037898, upper bound: 0.3029731
time: 2.64 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.46 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.46
Output dim: 1, lower bound: -0.3029732, upper bound: 0.3037898
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.46
Output dim: 1, lower bound: -0.3029735, upper bound: 0.3029740
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.46
Output dim: 1, lower bound: -0.3029741, upper bound: 0.3029734
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.46
Output dim: 1, lower bound: -0.3037898, upper bound: 0.3029731

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8729253, 0.8717177
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7148510, 0.7165015
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5730374, 0.5759881
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5075315, 0.5070286
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5516775, 0.5537603
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5274417, 0.5266745
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8956940, 0.8930349
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7361507, 0.7382290
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4861174, 0.4838424
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8189664, 0.8173366

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3029728, upper bound: 0.2979812
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2971647, upper bound: 0.3037893
time: 2.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8726478, 0.8719065
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7151464, 0.7160673
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5734756, 0.5753424
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5072885, 0.5071943
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5519458, 0.5533655
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5273440, 0.5267409
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8952367, 0.8933468
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7364168, 0.7378371
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4858929, 0.4839951
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8184457, 0.8176906

Time for backsubstitution: 22.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3022311, upper bound: 0.3029533
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3029528, upper bound: 0.3022317
time: 2.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8719950, 0.8726478
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7160672, 0.7152855
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5753424, 0.5736817
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5072711, 0.5072884
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5533655, 0.5520717
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5267720, 0.5273440
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8934922, 0.8952367
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7378368, 0.7365425
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4840671, 0.4858928
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8178554, 0.8184454

Time for backsubstitution: 23.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3022318, upper bound: 0.3029527
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3029534, upper bound: 0.3022311
time: 2.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8717179, 0.8728368
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7163626, 0.7148511
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5757806, 0.5730374
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5070287, 0.5074540
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5536339, 0.5516775
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5266745, 0.5274104
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8930349, 0.8955486
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7381029, 0.7361505
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4838425, 0.4860455
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8173366, 0.8187997

Time for backsubstitution: 23.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3030475, upper bound: 0.3029523
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3037692, upper bound: 0.3022307
time: 2.76 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.77 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.77
Output dim: 1, lower bound: -0.3029728, upper bound: 0.2979812
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.77
Output dim: 1, lower bound: -0.2971647, upper bound: 0.3037893
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.77
Output dim: 1, lower bound: -0.3022311, upper bound: 0.3029533
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.77
Output dim: 1, lower bound: -0.3029528, upper bound: 0.3022317
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.77
Output dim: 1, lower bound: -0.3022318, upper bound: 0.3029527
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.77
Output dim: 1, lower bound: -0.3029534, upper bound: 0.3022311
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.77
Output dim: 1, lower bound: -0.3030475, upper bound: 0.3029523
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.77
Output dim: 1, lower bound: -0.3037692, upper bound: 0.3022307

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8729634, 0.8717508
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7148651, 0.7165177
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5730387, 0.5759897
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5075423, 0.5070379
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5516927, 0.5537781
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5274491, 0.5266838
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8957272, 0.8930736
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7361655, 0.7382419
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4861051, 0.4838315
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8189776, 0.8173461

Time for backsubstitution: 23.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3022305, upper bound: 0.2979606
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3029521, upper bound: 0.2972389
time: 2.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8729582, 0.8717561
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7148672, 0.7165155
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5730389, 0.5759895
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5075408, 0.5070394
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5516952, 0.5537755
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5274510, 0.5266823
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8957324, 0.8930681
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7361631, 0.7382441
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4861063, 0.4838301
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8189757, 0.8173475

Time for backsubstitution: 23.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964224, upper bound: 0.3037688
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2971440, upper bound: 0.3030469
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8712211, 0.8736367
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7141994, 0.7172334
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5737222, 0.5751399
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5082762, 0.5063825
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5558754, 0.5501518
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5268397, 0.5273467
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8960385, 0.8926871
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7358379, 0.7385411
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4858373, 0.4840667
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8195846, 0.8167582

Time for backsubstitution: 23.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3022308, upper bound: 0.2971448
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964226, upper bound: 0.3029529
time: 2.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8726478, 0.8704798
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7151464, 0.7151202
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5732733, 0.5753424
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5064765, 0.5071943
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5487323, 0.5533655
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5273440, 0.5262367
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8945770, 0.8933468
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7364168, 0.7372577
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4858929, 0.4839396
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8175128, 0.8176906

Time for backsubstitution: 23.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3029524, upper bound: 0.2964231
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2971442, upper bound: 0.3022313
time: 2.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8705683, 0.8743780
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7151202, 0.7164516
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5755888, 0.5734792
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5082588, 0.5064765
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5572951, 0.5488580
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5262678, 0.5279498
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8942940, 0.8945770
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7372575, 0.7372468
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4840115, 0.4859643
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8189948, 0.8175130

Time for backsubstitution: 23.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3022314, upper bound: 0.2971441
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2964233, upper bound: 0.3029524
time: 2.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8719950, 0.8712211
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7160672, 0.7143384
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5751399, 0.5736817
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5064591, 0.5072884
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5501518, 0.5520717
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5267720, 0.5268397
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8928325, 0.8952367
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7378368, 0.7359633
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4840671, 0.4858375
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8169229, 0.8184454

Time for backsubstitution: 23.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3029530, upper bound: 0.2964225
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2971449, upper bound: 0.3022306
time: 2.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8702912, 0.8745668
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7154156, 0.7160172
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5760273, 0.5728348
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5080165, 0.5066422
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5575635, 0.5484638
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5261703, 0.5280162
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8938370, 0.8948889
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7375245, 0.7368546
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4837871, 0.4861171
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8184755, 0.8178670

Time for backsubstitution: 23.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3030471, upper bound: 0.2971439
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2972390, upper bound: 0.3029520
time: 2.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8717179, 0.8714101
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7163626, 0.7139039
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5755781, 0.5730374
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5062168, 0.5074540
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5504203, 0.5516775
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5266745, 0.5269061
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8923752, 0.8955486
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7381029, 0.7355714
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4838425, 0.4859900
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8164041, 0.8187997

Time for backsubstitution: 23.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3037688, upper bound: 0.2964221
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2979606, upper bound: 0.3022303
time: 2.78 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.85 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.85
Output dim: 1, lower bound: -0.3022305, upper bound: 0.2979606
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.85
Output dim: 1, lower bound: -0.3029521, upper bound: 0.2972389
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.85
Output dim: 1, lower bound: -0.2964224, upper bound: 0.3037688
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.85
Output dim: 1, lower bound: -0.2971440, upper bound: 0.3030469
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.85
Output dim: 1, lower bound: -0.3022308, upper bound: 0.2971448
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.85
Output dim: 1, lower bound: -0.2964226, upper bound: 0.3029529
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.85
Output dim: 1, lower bound: -0.3029524, upper bound: 0.2964231
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.85
Output dim: 1, lower bound: -0.2971442, upper bound: 0.3022313
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.85
Output dim: 1, lower bound: -0.3022314, upper bound: 0.2971441
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.85
Output dim: 1, lower bound: -0.2964233, upper bound: 0.3029524
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.85
Output dim: 1, lower bound: -0.3029530, upper bound: 0.2964225
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.85
Output dim: 1, lower bound: -0.2971449, upper bound: 0.3022306
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.85
Output dim: 1, lower bound: -0.3030471, upper bound: 0.2971439
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.85
Output dim: 1, lower bound: -0.2972390, upper bound: 0.3029520
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.85
Output dim: 1, lower bound: -0.3037688, upper bound: 0.2964221
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.85
Output dim: 1, lower bound: -0.2979606, upper bound: 0.3022303

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8715367, 0.8734808
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7139176, 0.7176836
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5732858, 0.5757877
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5085301, 0.5062260
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5556223, 0.5505645
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5269451, 0.5272896
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8965297, 0.8924141
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7355866, 0.7389462
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4860499, 0.4839033
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8201168, 0.8164136

Time for backsubstitution: 22.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2567
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 576
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 1409
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1261
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 2454
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1781

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2137

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2982967, upper bound: 0.2877271
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2921998, upper bound: 0.2941274
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8729634, 0.8703239
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7148651, 0.7155702
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5728366, 0.5759897
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5067304, 0.5070379
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5484792, 0.5537781
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5274491, 0.5261796
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8950679, 0.8930736
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7361655, 0.7376628
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4861051, 0.4837763
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8180454, 0.8173461

Time for backsubstitution: 22.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 1781
type: DSZ, layer: 3, pos: 2567
type: DSZ, layer: 3, pos: 1409
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 576
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2454
type: DSZ, layer: 3, pos: 1261
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1850

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2327

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3029160, upper bound: 0.2964841
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.3022370, upper bound: 0.2972025
time: 2.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8715310, 0.8734860
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7139198, 0.7176814
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5732861, 0.5757874
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5085286, 0.5062275
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5556248, 0.5505621
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5269465, 0.5272881
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8965349, 0.8924088
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7355847, 0.7389483
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4860513, 0.4839020
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8201149, 0.8164151

Time for backsubstitution: 22.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1781
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2454
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 1261
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 2567
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 1409
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 576
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 2474

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1781

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2868495, upper bound: 0.2943594
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2870140, upper bound: 0.2941955
time: 2.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8729582, 0.8703291
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7148672, 0.7155681
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5728369, 0.5759895
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5067289, 0.5070394
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5484817, 0.5537755
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5274510, 0.5261780
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8950734, 0.8930681
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7361631, 0.7376652
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4861063, 0.4837750
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8180435, 0.8173475

Time for backsubstitution: 23.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1261
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 1781
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2567
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 576
type: DSZ, layer: 3, pos: 1409
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1832
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2454
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1922

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1261

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.2921605, upper bound: 0.3030386
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2971358, upper bound: 0.2980587
time: 2.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.4397488, -7.1719198, -8.4397488, -7.1719198, -0.8712592, 0.8736699
1: 2.3758993, 3.4053402, 2.3758993, 3.4053402, -0.7142130, 0.7172494
2: -5.1056080, -4.1367159, -5.1056080, -4.1367159, -0.5737243, 0.5751419
3: -9.8951368, -8.8055744, -9.8951368, -8.8055744, -0.5082871, 0.5063916
4: -4.4544101, -3.5803537, -4.4544101, -3.5803537, -0.5558909, 0.5501698
5: -8.1140099, -7.1390996, -8.1140099, -7.1390996, -0.5268474, 0.5273558
6: -5.5528641, -4.3291721, -5.5528641, -4.3291721, -0.8960724, 0.8927255
7: -3.9400830, -3.0847538, -3.9400830, -3.0847538, -0.7358537, 0.7385540
8: -3.4837375, -2.6097441, -3.4837375, -2.6097441, -0.4858254, 0.4840561
9: -10.7434769, -9.5567389, -10.7434769, -9.5567389, -0.8195961, 0.8167677

Time for backsubstitution: 23.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1261
type: DSZ, layer: 3, pos: 557
type: DSZ, layer: 3, pos: 1922
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2137
type: DSZ, layer: 3, pos: 1507
type: DSZ, layer: 3, pos: 2474
type: DSZ, layer: 3, pos: 1451
type: DSZ, layer: 3, pos: 2124
type: DSZ, layer: 3, pos: 1781
type: DSZ, layer: 3, pos: 772
type: DSZ, layer: 3, pos: 576
type: DSZ, layer: 3, pos: 1839
type: DSZ, layer: 3, pos: 1409
type: DSZ, layer: 3, pos: 1851
type: DSZ, layer: 3, pos: 2559
type: DSZ, layer: 3, pos: 2327
type: DSZ, layer: 3, pos: 2340
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2454
type: DSZ, layer: 3, pos: 226
type: DSZ, layer: 3, pos: 2567
type: DSZ, layer: 3, pos: 1832

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2954129, upper bound: 0.2909915
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.2960775, upper bound: 0.2903261
time: 2.80 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 28.96 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.96
Output dim: 1, lower bound: -0.2982967, upper bound: 0.2877271
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.96
Output dim: 1, lower bound: -0.2921998, upper bound: 0.2941274
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.96
Output dim: 1, lower bound: -0.3029160, upper bound: 0.2964841
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.96
Output dim: 1, lower bound: -0.3022370, upper bound: 0.2972025
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.96
Output dim: 1, lower bound: -0.2868495, upper bound: 0.2943594
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.96
Output dim: 1, lower bound: -0.2870140, upper bound: 0.2941955
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.96
Output dim: 1, lower bound: -0.2921605, upper bound: 0.3030386
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.96
Output dim: 1, lower bound: -0.2971358, upper bound: 0.2980587
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.96
Output dim: 1, lower bound: -0.2954129, upper bound: 0.2909915
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.96
Output dim: 1, lower bound: -0.2960775, upper bound: 0.2903261
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 1, lower bound: -0.2964226, upper bound: 0.3029529
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 1, lower bound: -0.3029524, upper bound: 0.2964231
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 1, lower bound: -0.2971442, upper bound: 0.3022313
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 1, lower bound: -0.3022314, upper bound: 0.2971441
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 1, lower bound: -0.2964233, upper bound: 0.3029524
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 1, lower bound: -0.3029530, upper bound: 0.2964225
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 1, lower bound: -0.2971449, upper bound: 0.3022306
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 1, lower bound: -0.3030471, upper bound: 0.2971439
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 1, lower bound: -0.2972390, upper bound: 0.3029520
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 1, lower bound: -0.3037688, upper bound: 0.2964221
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.96
Output dim: 1, lower bound: -0.2979606, upper bound: 0.3022303

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.93 + 548.60 = 605.53 seconds
