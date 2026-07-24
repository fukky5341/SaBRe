## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.46374671450000005


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7471986, 0.7471986)
1: (-8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.9235189, 0.9235187)
2: (-2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7697604, 0.7697604)
3: (-10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8960600, 0.8960595)
4: (-8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.8202255, 0.8202257)
5: (-5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6728578, 0.6728578)
6: (-1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.8079462, 0.8079464)
7: (-8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.9045877, 0.9045880)
8: (-1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7171779, 0.7171779)
9: (-6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.8136141, 0.8136144)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.37 + 32.71 = 56.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.4660768, upper bound: 0.4660770

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 6126
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 541

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4652693, upper bound: 0.4660761
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4660759, upper bound: 0.4652695
time: 2.86 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.11 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.11
Output dim: 0, lower bound: -0.4652693, upper bound: 0.4660761
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.11
Output dim: 0, lower bound: -0.4660759, upper bound: 0.4652695

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7460160, 0.7476714
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.9206057, 0.9195912
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7704310, 0.7709532
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8964496, 0.8955588
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.8201234, 0.8194458
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6697938, 0.6687577
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.8099246, 0.8106079
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8974268, 0.8992150
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7136383, 0.7124586
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7966850, 0.8009104

Time for backsubstitution: 21.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 6126
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 495

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4642396, upper bound: 0.4660589
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4652520, upper bound: 0.4650464
time: 2.99 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7476714, 0.7460160
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.9195910, 0.9206059
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7709532, 0.7704313
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8955588, 0.8964496
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.8194463, 0.8201237
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6687577, 0.6697941
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.8106079, 0.8099246
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8992150, 0.8974268
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7124586, 0.7136383
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.8009105, 0.7966850

Time for backsubstitution: 22.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 6126
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 495

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4650462, upper bound: 0.4652523
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4660587, upper bound: 0.4642397
time: 3.04 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.34 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.34
Output dim: 0, lower bound: -0.4642396, upper bound: 0.4660589
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.34
Output dim: 0, lower bound: -0.4652520, upper bound: 0.4650464
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.34
Output dim: 0, lower bound: -0.4650462, upper bound: 0.4652523
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.34
Output dim: 0, lower bound: -0.4660587, upper bound: 0.4642397

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7450888, 0.7514553
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.9238634, 0.9187932
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7722931, 0.7704976
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8955250, 0.8993323
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.8210187, 0.8192267
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6711910, 0.6684163
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.8134379, 0.8097486
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8966224, 0.9025104
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7142520, 0.7123089
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7976551, 0.8006729

Time for backsubstitution: 22.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6126
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 6126

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4642319, upper bound: 0.4605960
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4587767, upper bound: 0.4660512
time: 3.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7460160, 0.7467442
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.9198074, 0.9195912
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7699757, 0.7709532
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8964496, 0.8946340
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.8199043, 0.8194458
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6694524, 0.6687577
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.8090653, 0.8106079
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8974268, 0.8984106
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7134881, 0.7124586
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7964475, 0.8009104

Time for backsubstitution: 22.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6126
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 6126

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4652443, upper bound: 0.4595836
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4597892, upper bound: 0.4650387
time: 2.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7467442, 0.7498002
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.9228487, 0.9198079
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7728152, 0.7699757
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8946338, 0.9002233
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.8203411, 0.8199046
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6701543, 0.6694527
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.8141208, 0.8090653
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8984106, 0.9007225
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7130723, 0.7134883
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.8018806, 0.7964475

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6126
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 6126

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4650385, upper bound: 0.4597893
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4595834, upper bound: 0.4652446
time: 2.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7476714, 0.7450888
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.9187927, 0.9206059
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7704978, 0.7704313
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8955588, 0.8955250
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.8192267, 0.8201237
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6684158, 0.6697941
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.8097486, 0.8099246
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8992150, 0.8966224
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7123089, 0.7136383
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.8006728, 0.7966850

Time for backsubstitution: 22.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6126
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 6126

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4660510, upper bound: 0.4587769
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4605958, upper bound: 0.4642320
time: 3.32 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.54 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.54
Output dim: 0, lower bound: -0.4642319, upper bound: 0.4605960
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.54
Output dim: 0, lower bound: -0.4587767, upper bound: 0.4660512
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.54
Output dim: 0, lower bound: -0.4652443, upper bound: 0.4595836
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.54
Output dim: 0, lower bound: -0.4597892, upper bound: 0.4650387
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.54
Output dim: 0, lower bound: -0.4650385, upper bound: 0.4597893
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.54
Output dim: 0, lower bound: -0.4595834, upper bound: 0.4652446
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.54
Output dim: 0, lower bound: -0.4660510, upper bound: 0.4587769
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.54
Output dim: 0, lower bound: -0.4605958, upper bound: 0.4642320

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7334490, 0.7359385
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8599632, 0.8708379
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7713249, 0.7692077
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8445907, 0.8314486
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7558811, 0.7703600
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6718473, 0.6644137
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7974067, 0.8002689
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8968477, 0.9026942
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7131252, 0.7138925
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7386715, 0.7220652

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 850

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4642226, upper bound: 0.4484210
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4520568, upper bound: 0.4605868
time: 3.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7295718, 0.7398155
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8759079, 0.8548932
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7710032, 0.7695293
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8276415, 0.8483982
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7721519, 0.7540889
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6671886, 0.6690724
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.8039584, 0.7937171
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8968060, 0.9027359
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7158356, 0.7111821
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7190473, 0.7416893

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 850

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4587675, upper bound: 0.4538762
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4466017, upper bound: 0.4660420
time: 2.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7343757, 0.7312272
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8559077, 0.8716359
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7690074, 0.7696631
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8455153, 0.8267503
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7547667, 0.7705791
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6701088, 0.6647551
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7930341, 0.8011284
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8976524, 0.8985941
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7123618, 0.7140424
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7374640, 0.7223029

Time for backsubstitution: 22.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 850

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4652351, upper bound: 0.4474086
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4530693, upper bound: 0.4595744
time: 2.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7304990, 0.7351041
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8718526, 0.8556912
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7686856, 0.7699847
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8285661, 0.8436995
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7710376, 0.7543080
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6654501, 0.6694138
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7995858, 0.7945764
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8976107, 0.8986359
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7150722, 0.7113321
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7178400, 0.7419269

Time for backsubstitution: 22.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 850

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4597799, upper bound: 0.4528637
time: 3.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4476141, upper bound: 0.4650295
time: 2.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7351041, 0.7342832
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8589485, 0.8718524
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7718468, 0.7686858
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8436995, 0.8323398
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7552035, 0.7710376
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6708107, 0.6654501
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7980895, 0.7995858
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8986359, 0.9009063
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7119460, 0.7150719
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7428968, 0.7178400

Time for backsubstitution: 22.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 850

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4650293, upper bound: 0.4476143
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4528635, upper bound: 0.4597802
time: 2.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7312272, 0.7381601
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8748934, 0.8559077
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7715251, 0.7690072
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8267503, 0.8492889
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7714741, 0.7547667
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6661520, 0.6701088
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.8046412, 0.7930338
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8985941, 0.9009480
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7146564, 0.7123616
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7232728, 0.7374638

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 850

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4595742, upper bound: 0.4530695
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4474084, upper bound: 0.4652353
time: 3.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7360311, 0.7295718
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8548930, 0.8726506
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7695293, 0.7691412
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8446245, 0.8276415
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7540891, 0.7712567
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6690722, 0.6657917
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7937169, 0.8004453
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8994405, 0.8968060
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7111821, 0.7152219
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7416892, 0.7180775

Time for backsubstitution: 22.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 850

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4660418, upper bound: 0.4466018
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4538760, upper bound: 0.4587677
time: 2.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7321541, 0.7334490
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8708379, 0.8567057
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7692077, 0.7694628
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8276753, 0.8445907
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7703598, 0.7549858
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6644135, 0.6704504
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.8002687, 0.7938933
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8993988, 0.8968480
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7138925, 0.7125115
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7220652, 0.7377014

Time for backsubstitution: 22.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 850

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4605866, upper bound: 0.4520569
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4484208, upper bound: 0.4642229
time: 3.10 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.60 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.60
Output dim: 0, lower bound: -0.4642226, upper bound: 0.4484210
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 0, lower bound: -0.4520568, upper bound: 0.4605868
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 0, lower bound: -0.4587675, upper bound: 0.4538762
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.60
Output dim: 0, lower bound: -0.4466017, upper bound: 0.4660420
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.60
Output dim: 0, lower bound: -0.4652351, upper bound: 0.4474086
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 0, lower bound: -0.4530693, upper bound: 0.4595744
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 0, lower bound: -0.4597799, upper bound: 0.4528637
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.60
Output dim: 0, lower bound: -0.4476141, upper bound: 0.4650295
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.60
Output dim: 0, lower bound: -0.4650293, upper bound: 0.4476143
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 0, lower bound: -0.4528635, upper bound: 0.4597802
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 0, lower bound: -0.4595742, upper bound: 0.4530695
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.60
Output dim: 0, lower bound: -0.4474084, upper bound: 0.4652353
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.60
Output dim: 0, lower bound: -0.4660418, upper bound: 0.4466018
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 0, lower bound: -0.4538760, upper bound: 0.4587677
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.60
Output dim: 0, lower bound: -0.4605866, upper bound: 0.4520569
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.60
Output dim: 0, lower bound: -0.4484208, upper bound: 0.4642229

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7390885, 0.7400374
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8534830, 0.8618219
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7666364, 0.7629621
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8380570, 0.8266041
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7569394, 0.7697933
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6637423, 0.6583269
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7926292, 0.7966859
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8960154, 0.9015844
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7140517, 0.7150602
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7414351, 0.7241024

Time for backsubstitution: 22.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4572165, upper bound: 0.4481598
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4639615, upper bound: 0.4414149
time: 3.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7336707, 0.7454553
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8668919, 0.8484130
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7647574, 0.7648408
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8227963, 0.8418648
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7715852, 0.7551475
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6611016, 0.6609678
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.8003750, 0.7889402
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8956962, 0.9019034
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7170033, 0.7121081
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7210846, 0.7444530

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4395955, upper bound: 0.4657808
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4463405, upper bound: 0.4590357
time: 3.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7400150, 0.7353263
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8494275, 0.8626206
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7643189, 0.7634177
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8389821, 0.8219059
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7558250, 0.7700126
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6620042, 0.6586688
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7882566, 0.7975454
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8968201, 0.8974843
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7132878, 0.7152109
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7402275, 0.7243395

Time for backsubstitution: 22.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4582289, upper bound: 0.4471474
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4649739, upper bound: 0.4404024
time: 3.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7345972, 0.7407441
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8628366, 0.8492117
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7624400, 0.7652967
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8237219, 0.8371663
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7704709, 0.7553666
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6593630, 0.6613097
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7960033, 0.7897997
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8965008, 0.8978035
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7162395, 0.7122591
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7198770, 0.7446901

Time for backsubstitution: 22.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4406080, upper bound: 0.4647683
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4473530, upper bound: 0.4580232
time: 3.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 6.1240320, 7.4177628, 6.1240320, 7.4177628, -0.7407439, 0.7383823
1: -8.8166962, -7.1378717, -8.8166962, -7.1378717, -0.8524683, 0.8628366
2: -2.9785883, -1.6955307, -2.9785883, -1.6955307, -0.7671583, 0.7624400
3: -10.3806295, -9.0514908, -10.3806295, -9.0514908, -0.8371663, 0.8274953
4: -8.3440456, -6.9297085, -8.3440456, -6.9297085, -0.7562616, 0.7704711
5: -5.8682699, -4.9259844, -5.8682699, -4.9259844, -0.6627061, 0.6593633
6: -1.6049871, -0.3183823, -1.6049871, -0.3183823, -0.7933121, 0.7960029
7: -8.5092411, -6.7643943, -8.5092411, -6.7643943, -0.8978033, 0.8997965
8: -1.6987939, -0.7250729, -1.6987939, -0.7250729, -0.7128720, 0.7162397
9: -6.3969994, -4.8874454, -6.3969994, -4.8874454, -0.7456605, 0.7198770

Time for backsubstitution: 22.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 136

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4580231, upper bound: 0.4473531
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.4647681, upper bound: 0.4406081
time: 3.15 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 28.88 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.88
Output dim: 0, lower bound: -0.4572165, upper bound: 0.4481598
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.88
Output dim: 0, lower bound: -0.4639615, upper bound: 0.4414149
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.88
Output dim: 0, lower bound: -0.4395955, upper bound: 0.4657808
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.88
Output dim: 0, lower bound: -0.4463405, upper bound: 0.4590357
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.88
Output dim: 0, lower bound: -0.4582289, upper bound: 0.4471474
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.88
Output dim: 0, lower bound: -0.4649739, upper bound: 0.4404024
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.88
Output dim: 0, lower bound: -0.4406080, upper bound: 0.4647683
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.88
Output dim: 0, lower bound: -0.4473530, upper bound: 0.4580232
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.88
Output dim: 0, lower bound: -0.4580231, upper bound: 0.4473531
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.88
Output dim: 0, lower bound: -0.4647681, upper bound: 0.4406081
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.88
Output dim: 0, lower bound: -0.4474084, upper bound: 0.4652353
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.88
Output dim: 0, lower bound: -0.4660418, upper bound: 0.4466018
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.88
Output dim: 0, lower bound: -0.4484208, upper bound: 0.4642229

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.08 + 548.75 = 604.82 seconds
