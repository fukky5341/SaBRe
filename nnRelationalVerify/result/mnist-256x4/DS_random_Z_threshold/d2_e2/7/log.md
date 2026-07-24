## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0134062


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010673, 0.0010673)
1: (-0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0059097, 0.0059097)
2: (0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0132030, 0.0132030)
3: (-0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0055638, 0.0055638)
4: (0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0215853, 0.0215853)
5: (0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0041992, 0.0041992)
6: (-0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0054646, 0.0054646)
7: (-0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006971, 0.0006971)
8: (-0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0037756, 0.0037756)
9: (-0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0189017, 0.0189017)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.76 + 2.36 = 3.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0157720, upper bound: 0.0157720

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0155331, upper bound: 0.0155394
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0155394, upper bound: 0.0155331
time: 1.35 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.80 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.80
Output dim: 4, lower bound: -0.0155331, upper bound: 0.0155394
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.80
Output dim: 4, lower bound: -0.0155394, upper bound: 0.0155331

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010465, 0.0010480
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0058027, 0.0057942
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0129450, 0.0129638
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0054630, 0.0054550
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0211942, 0.0211635
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0041231, 0.0041171
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0053578, 0.0053656
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006834, 0.0006844
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0037072, 0.0037018
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0185323, 0.0185592

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 218

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0153702, upper bound: 0.0151370
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0151204, upper bound: 0.0153745
time: 1.43 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010480, 0.0010465
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0057942, 0.0058027
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0129638, 0.0129450
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0054550, 0.0054630
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0211635, 0.0211942
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0041171, 0.0041231
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0053656, 0.0053578
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006844, 0.0006834
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0037018, 0.0037072
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0185592, 0.0185323

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0155127, upper bound: 0.0152103
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0152228, upper bound: 0.0155063
time: 1.41 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.51 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.51
Output dim: 4, lower bound: -0.0153702, upper bound: 0.0151370
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.51
Output dim: 4, lower bound: -0.0151204, upper bound: 0.0153745
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.51
Output dim: 4, lower bound: -0.0155127, upper bound: 0.0152103
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.51
Output dim: 4, lower bound: -0.0152228, upper bound: 0.0155063

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010334, 0.0010441
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0057810, 0.0057217
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0127829, 0.0129153
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0054425, 0.0053868
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0211150, 0.0208985
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0041077, 0.0040656
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052908, 0.0053456
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006749, 0.0006819
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036934, 0.0036555
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0183003, 0.0184899

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149440, upper bound: 0.0148918
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0151245, upper bound: 0.0147222
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010465, 0.0010349
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0057301, 0.0057942
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0129450, 0.0128017
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053947, 0.0054550
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0209293, 0.0211635
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040716, 0.0041171
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0053578, 0.0052986
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006834, 0.0006759
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036609, 0.0037018
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0185323, 0.0183273

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139637, upper bound: 0.0141493
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139637, upper bound: 0.0141493
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010360, 0.0010399
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0057578, 0.0057361
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0128152, 0.0128635
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0054207, 0.0054003
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0210303, 0.0209513
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040912, 0.0040759
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0053041, 0.0053241
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006766, 0.0006791
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036785, 0.0036647
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0183465, 0.0184157

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0155053, upper bound: 0.0147869
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150058, upper bound: 0.0152012
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010417, 0.0010345
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0057277, 0.0057680
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0128863, 0.0127964
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053924, 0.0054303
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0209205, 0.0210676
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040699, 0.0040985
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0053336, 0.0052963
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006803, 0.0006756
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036593, 0.0036851
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0184484, 0.0183196

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0144782, upper bound: 0.0148057
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0145576, upper bound: 0.0147728
time: 1.38 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.41 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 4, lower bound: -0.0149440, upper bound: 0.0148918
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 4, lower bound: -0.0151245, upper bound: 0.0147222
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 4, lower bound: -0.0139637, upper bound: 0.0141493
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 4, lower bound: -0.0139637, upper bound: 0.0141493
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 4, lower bound: -0.0155053, upper bound: 0.0147869
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 4, lower bound: -0.0150058, upper bound: 0.0152012
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 4, lower bound: -0.0144782, upper bound: 0.0148057
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.41
Output dim: 4, lower bound: -0.0145576, upper bound: 0.0147728

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010103, 0.0010168
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0056298, 0.0055939
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0124973, 0.0125776
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053002, 0.0052664
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0205628, 0.0204316
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040003, 0.0039748
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0051726, 0.0052058
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006598, 0.0006640
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035968, 0.0035738
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0178914, 0.0180063

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0137622, upper bound: 0.0137491
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0137622, upper bound: 0.0137491
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010061, 0.0010216
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0056566, 0.0055705
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0124452, 0.0126376
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053255, 0.0052444
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0206609, 0.0203464
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040194, 0.0039582
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0051510, 0.0052306
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006571, 0.0006672
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036139, 0.0035589
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0178168, 0.0180922

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0140050, upper bound: 0.0137582
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0140050, upper bound: 0.0137582
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010447, 0.0010268
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0056851, 0.0057844
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0129229, 0.0127011
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053523, 0.0054458
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0207648, 0.0211274
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040396, 0.0041101
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0053487, 0.0052569
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006823, 0.0006706
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036321, 0.0036955
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0185008, 0.0181832

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0104183, upper bound: 0.0104359
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0104183, upper bound: 0.0104359
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010383, 0.0010349
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0057301, 0.0057492
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0128444, 0.0128017
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053947, 0.0054126
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0209293, 0.0209990
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040716, 0.0040851
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0053162, 0.0052986
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006781, 0.0006759
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036609, 0.0036731
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0183883, 0.0183273

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0135335, upper bound: 0.0136976
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0135239, upper bound: 0.0137215
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010320, 0.0010574
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0058546, 0.0057142
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0127662, 0.0130798
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0055118, 0.0053797
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0213838, 0.0208712
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0041600, 0.0040603
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052839, 0.0054136
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006740, 0.0006906
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0037404, 0.0036507
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0182764, 0.0187253

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0152538, upper bound: 0.0144144
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0151606, upper bound: 0.0145614
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010534, 0.0010359
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0057359, 0.0058325
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0130305, 0.0128145
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0054001, 0.0054911
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0209502, 0.0213034
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040757, 0.0041444
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0053933, 0.0053039
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006880, 0.0006766
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036645, 0.0037263
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0186548, 0.0183456

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0134841, upper bound: 0.0134978
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0134841, upper bound: 0.0134978
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010236, 0.0010109
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0055973, 0.0056678
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0126626, 0.0125051
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052697, 0.0053360
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0204443, 0.0207018
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039772, 0.0040273
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052410, 0.0051758
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006685, 0.0006602
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035760, 0.0036211
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0181280, 0.0179025

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0142190, upper bound: 0.0144781
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0142121, upper bound: 0.0144917
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010182, 0.0010345
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0057277, 0.0056376
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0125950, 0.0127964
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053924, 0.0053076
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0209205, 0.0205914
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040699, 0.0040058
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052130, 0.0052963
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006650, 0.0006756
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036593, 0.0036018
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0180313, 0.0183196

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0133544, upper bound: 0.0135803
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0133544, upper bound: 0.0135803
time: 1.07 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.84 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0137622, upper bound: 0.0137491
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0137622, upper bound: 0.0137491
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0140050, upper bound: 0.0137582
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0140050, upper bound: 0.0137582
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0104183, upper bound: 0.0104359
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0104183, upper bound: 0.0104359
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0135335, upper bound: 0.0136976
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0135239, upper bound: 0.0137215
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0152538, upper bound: 0.0144144
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0151606, upper bound: 0.0145614
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0134841, upper bound: 0.0134978
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0134841, upper bound: 0.0134978
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0142190, upper bound: 0.0144781
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0142121, upper bound: 0.0144917
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0133544, upper bound: 0.0135803
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.84
Output dim: 4, lower bound: -0.0133544, upper bound: 0.0135803

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010094, 0.0010088
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0055858, 0.0055892
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0124869, 0.0124793
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052588, 0.0052620
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0204022, 0.0204146
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039690, 0.0039715
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0051683, 0.0051651
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006593, 0.0006589
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035687, 0.0035708
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0178766, 0.0178657

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0121349, upper bound: 0.0121423
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0121349, upper bound: 0.0121423
time: 0.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010023, 0.0010168
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0056298, 0.0055499
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0123990, 0.0125776
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053002, 0.0052250
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0205628, 0.0202709
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040003, 0.0039435
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0051319, 0.0052058
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006546, 0.0006640
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035968, 0.0035457
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0177508, 0.0180063

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0133512, upper bound: 0.0133061
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0133512, upper bound: 0.0133061
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010059, 0.0010216
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0056563, 0.0055699
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0124438, 0.0126368
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053252, 0.0052438
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0206597, 0.0203441
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040191, 0.0039577
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0051504, 0.0052303
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006570, 0.0006672
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036137, 0.0035585
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0178148, 0.0180912

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0138267, upper bound: 0.0136812
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139255, upper bound: 0.0135831
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010061, 0.0010215
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0056560, 0.0055705
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0124452, 0.0126362
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053249, 0.0052444
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0206586, 0.0203464
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040189, 0.0039582
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0051510, 0.0052300
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006571, 0.0006671
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036135, 0.0035589
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0178168, 0.0180902

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139784, upper bound: 0.0135276
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0137599, upper bound: 0.0137306
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010339, 0.0010326
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0057177, 0.0057247
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0127896, 0.0127740
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053830, 0.0053896
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0208840, 0.0209094
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040628, 0.0040677
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052935, 0.0052871
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006752, 0.0006744
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036530, 0.0036574
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0183099, 0.0182876

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0134252, upper bound: 0.0136349
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0134729, upper bound: 0.0135771
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010383, 0.0010304
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0057055, 0.0057492
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0128444, 0.0127468
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053715, 0.0054126
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0208394, 0.0209990
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040541, 0.0040851
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0053162, 0.0052758
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006781, 0.0006730
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036452, 0.0036731
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0183883, 0.0182486

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0123271, upper bound: 0.0124666
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0123271, upper bound: 0.0124666
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010274, 0.0010542
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0058371, 0.0056885
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0127087, 0.0130406
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0054954, 0.0053555
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0213199, 0.0207772
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0041476, 0.0040420
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052601, 0.0053974
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006710, 0.0006885
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0037292, 0.0036343
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0181941, 0.0186693

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0144306, upper bound: 0.0136884
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0144306, upper bound: 0.0136884
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010320, 0.0010527
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0058288, 0.0057142
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0127662, 0.0130223
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0054876, 0.0053797
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0212898, 0.0208712
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0041417, 0.0040603
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052839, 0.0053898
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006740, 0.0006875
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0037239, 0.0036507
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0182764, 0.0186430

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149007, upper bound: 0.0143276
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149007, upper bound: 0.0143276
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010487, 0.0010410
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0057639, 0.0058064
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0129721, 0.0128772
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0054265, 0.0054665
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0210527, 0.0212078
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040956, 0.0041258
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0053691, 0.0053298
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006849, 0.0006799
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036825, 0.0037096
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0185711, 0.0184354

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0128855, upper bound: 0.0122669
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0121671, upper bound: 0.0128463
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010534, 0.0010312
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0057097, 0.0058325
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0130305, 0.0127561
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053754, 0.0054911
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0208546, 0.0213034
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040571, 0.0041444
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0053933, 0.0052797
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006880, 0.0006735
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036478, 0.0037263
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0186548, 0.0182619

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0124889, upper bound: 0.0125830
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125461, upper bound: 0.0124703
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010218, 0.0010101
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0055927, 0.0056576
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0126396, 0.0124947
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052653, 0.0053264
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0204273, 0.0206643
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039739, 0.0040200
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052315, 0.0051715
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006673, 0.0006597
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035731, 0.0036145
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0180952, 0.0178877

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139420, upper bound: 0.0141280
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0138436, upper bound: 0.0141876
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010236, 0.0010090
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0055871, 0.0056678
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0126626, 0.0124821
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052600, 0.0053360
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0204068, 0.0207018
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039699, 0.0040273
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052410, 0.0051663
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006685, 0.0006590
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035695, 0.0036211
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0181280, 0.0178697

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0115270, upper bound: 0.0116227
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0115270, upper bound: 0.0116227
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010126, 0.0010283
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0056939, 0.0056068
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0125263, 0.0127207
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053605, 0.0052786
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0207969, 0.0204790
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040458, 0.0039840
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0051846, 0.0052650
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006613, 0.0006716
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036377, 0.0035821
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0179329, 0.0182113

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0132089, upper bound: 0.0134945
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0132779, upper bound: 0.0134092
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010182, 0.0010289
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0056969, 0.0056376
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0125950, 0.0127276
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053634, 0.0053076
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0208081, 0.0205914
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040480, 0.0040058
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052130, 0.0052679
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006650, 0.0006720
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036397, 0.0036018
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0180313, 0.0182212

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0129025, upper bound: 0.0130639
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0129025, upper bound: 0.0130639
time: 0.99 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.51 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0121349, upper bound: 0.0121423
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0121349, upper bound: 0.0121423
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0133512, upper bound: 0.0133061
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0133512, upper bound: 0.0133061
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0138267, upper bound: 0.0136812
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0139255, upper bound: 0.0135831
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0139784, upper bound: 0.0135276
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0137599, upper bound: 0.0137306
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0134252, upper bound: 0.0136349
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0134729, upper bound: 0.0135771
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0123271, upper bound: 0.0124666
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0123271, upper bound: 0.0124666
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0144306, upper bound: 0.0136884
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0144306, upper bound: 0.0136884
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0149007, upper bound: 0.0143276
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0149007, upper bound: 0.0143276
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0128855, upper bound: 0.0122669
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0121671, upper bound: 0.0128463
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0124889, upper bound: 0.0125830
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0125461, upper bound: 0.0124703
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0139420, upper bound: 0.0141280
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0138436, upper bound: 0.0141876
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0115270, upper bound: 0.0116227
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0115270, upper bound: 0.0116227
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0132089, upper bound: 0.0134945
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0132779, upper bound: 0.0134092
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0129025, upper bound: 0.0130639
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.51
Output dim: 4, lower bound: -0.0129025, upper bound: 0.0130639

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009916, 0.0010026
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0055513, 0.0054906
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0122666, 0.0124022
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052263, 0.0051692
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0202761, 0.0200544
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039445, 0.0039014
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0050771, 0.0051332
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006476, 0.0006548
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035466, 0.0035078
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0175611, 0.0177553

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119180, upper bound: 0.0119038
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119180, upper bound: 0.0119038
time: 0.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009870, 0.0010061
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0055710, 0.0054649
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0122092, 0.0124461
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052448, 0.0051450
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0203479, 0.0199605
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039585, 0.0038831
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0050533, 0.0051514
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006446, 0.0006571
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035592, 0.0034914
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0174789, 0.0178182

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0139230, upper bound: 0.0132866
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0135704, upper bound: 0.0135765
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009942, 0.0010156
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0056234, 0.0055049
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0122986, 0.0125633
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052942, 0.0051826
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0205394, 0.0201067
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039957, 0.0039115
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0050903, 0.0051999
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006493, 0.0006633
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035927, 0.0035170
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0176069, 0.0179859

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0128312, upper bound: 0.0125053
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0128312, upper bound: 0.0125053
time: 0.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009996, 0.0010097
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0055904, 0.0055350
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0123657, 0.0124896
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052631, 0.0052109
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0204190, 0.0202164
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039723, 0.0039329
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0051181, 0.0051694
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006529, 0.0006594
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035716, 0.0035362
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0177030, 0.0178804

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0118738, upper bound: 0.0119638
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0118738, upper bound: 0.0119638
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010205, 0.0010149
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0056195, 0.0056505
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0126238, 0.0125545
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052905, 0.0053197
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0205251, 0.0206384
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039929, 0.0040150
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052249, 0.0051962
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006665, 0.0006628
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035902, 0.0036100
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0180726, 0.0179733

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 124

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0133943, upper bound: 0.0134106
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0132320, upper bound: 0.0136064
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010159, 0.0010190
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0056424, 0.0056248
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0125664, 0.0126058
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053121, 0.0052955
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0206090, 0.0205446
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040093, 0.0039967
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052012, 0.0052175
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006635, 0.0006655
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036048, 0.0035936
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0179904, 0.0180467

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0128122, upper bound: 0.0128918
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0127800, upper bound: 0.0129275
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010221, 0.0010487
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0058069, 0.0056593
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0126436, 0.0129732
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0054669, 0.0053280
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0212096, 0.0206708
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0041261, 0.0040213
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052331, 0.0053695
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006675, 0.0006849
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0037099, 0.0036157
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0181009, 0.0185727

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0126769, upper bound: 0.0121710
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0126769, upper bound: 0.0121710
time: 0.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010274, 0.0010489
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0058079, 0.0056885
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0127087, 0.0129755
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0054679, 0.0053555
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0212135, 0.0207772
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0041269, 0.0040420
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052601, 0.0053705
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006710, 0.0006851
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0037106, 0.0036343
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0181941, 0.0185761

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131795, upper bound: 0.0126371
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0131795, upper bound: 0.0126257
time: 1.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010210, 0.0010417
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0057677, 0.0056531
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0126297, 0.0128857
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0054301, 0.0053222
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0210667, 0.0206481
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040983, 0.0040169
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052274, 0.0053333
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006668, 0.0006803
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036849, 0.0036117
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0180810, 0.0184475

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0113659, upper bound: 0.0112150
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0113659, upper bound: 0.0112150
time: 0.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010210, 0.0010527
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0058288, 0.0056531
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0126297, 0.0130223
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0054876, 0.0053222
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0212898, 0.0206480
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0041417, 0.0040169
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052274, 0.0053898
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006668, 0.0006875
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0037239, 0.0036117
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0180810, 0.0186430

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0143288, upper bound: 0.0130529
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0134323, upper bound: 0.0136315
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010173, 0.0010077
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0055794, 0.0056327
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0125840, 0.0124649
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052528, 0.0053029
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0203787, 0.0205734
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039645, 0.0040023
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052085, 0.0051592
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006644, 0.0006581
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035646, 0.0035986
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0180156, 0.0178451

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0133921, upper bound: 0.0134636
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0132955, upper bound: 0.0135695
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010218, 0.0010056
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0055678, 0.0056576
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0126396, 0.0124391
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052419, 0.0053264
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0203364, 0.0206643
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039562, 0.0040200
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052315, 0.0051485
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006673, 0.0006567
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035572, 0.0036145
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0180952, 0.0178081

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 195

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0132168, upper bound: 0.0127696
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125053, upper bound: 0.0135693
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009997, 0.0010109
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0055973, 0.0055351
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0123660, 0.0125050
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052696, 0.0052111
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0204441, 0.0202170
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039772, 0.0039330
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0051182, 0.0051757
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006529, 0.0006602
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035760, 0.0035363
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0177035, 0.0179024

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0128397, upper bound: 0.0130901
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0128163, upper bound: 0.0130997
time: 1.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009953, 0.0010155
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0056229, 0.0055110
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0123122, 0.0125623
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052938, 0.0051884
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0205378, 0.0201289
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039954, 0.0039159
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0050959, 0.0051995
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006500, 0.0006632
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035924, 0.0035209
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0176264, 0.0179844

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 218

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0130214, upper bound: 0.0130317
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0129437, upper bound: 0.0131327
time: 1.07 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.58 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0119180, upper bound: 0.0119038
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0119180, upper bound: 0.0119038
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0139230, upper bound: 0.0132866
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0135704, upper bound: 0.0135765
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0128312, upper bound: 0.0125053
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0128312, upper bound: 0.0125053
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0118738, upper bound: 0.0119638
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0118738, upper bound: 0.0119638
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0133943, upper bound: 0.0134106
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0132320, upper bound: 0.0136064
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0128122, upper bound: 0.0128918
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0127800, upper bound: 0.0129275
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0126769, upper bound: 0.0121710
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0126769, upper bound: 0.0121710
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0131795, upper bound: 0.0126371
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0131795, upper bound: 0.0126257
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0113659, upper bound: 0.0112150
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0113659, upper bound: 0.0112150
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0143288, upper bound: 0.0130529
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0134323, upper bound: 0.0136315
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0133921, upper bound: 0.0134636
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0132955, upper bound: 0.0135695
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0132168, upper bound: 0.0127696
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0125053, upper bound: 0.0135693
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0128397, upper bound: 0.0130901
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0128163, upper bound: 0.0130997
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0130214, upper bound: 0.0130317
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.58
Output dim: 4, lower bound: -0.0129437, upper bound: 0.0131327

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009844, 0.0010250
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0056754, 0.0054507
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0121774, 0.0126796
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053432, 0.0051316
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0207296, 0.0199086
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040327, 0.0038730
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0050402, 0.0052480
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006429, 0.0006694
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036259, 0.0034823
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0174335, 0.0181524

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119703, upper bound: 0.0116331
time: 0.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119703, upper bound: 0.0116331
time: 0.90 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010060, 0.0010036
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0055567, 0.0055702
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0124444, 0.0124144
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052315, 0.0052441
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0202960, 0.0203450
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039484, 0.0039579
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0051507, 0.0051382
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006570, 0.0006554
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035501, 0.0035587
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0178157, 0.0177727

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0099907, upper bound: 0.0100028
time: 0.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0099907, upper bound: 0.0100028
time: 0.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010093, 0.0010092
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0055879, 0.0055884
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0124852, 0.0124841
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052608, 0.0052613
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0204100, 0.0204117
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039706, 0.0039709
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0051675, 0.0051671
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006592, 0.0006591
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035700, 0.0035703
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0178741, 0.0178725

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122038, upper bound: 0.0121882
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122038, upper bound: 0.0121882
time: 0.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010151, 0.0010034
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0055560, 0.0056208
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0125574, 0.0124128
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052308, 0.0052917
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0202934, 0.0205299
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039479, 0.0039939
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0051975, 0.0051376
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006630, 0.0006553
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035496, 0.0035910
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0179775, 0.0177704

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120408, upper bound: 0.0123677
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120408, upper bound: 0.0123677
time: 1.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009866, 0.0010355
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0057335, 0.0054629
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0122047, 0.0128093
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053979, 0.0051431
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0209416, 0.0199533
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040740, 0.0038817
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0050515, 0.0053017
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006444, 0.0006763
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036630, 0.0034901
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0174726, 0.0183381

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0143288, upper bound: 0.0130399
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0142830, upper bound: 0.0130529
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010210, 0.0010187
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0056402, 0.0056531
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0126297, 0.0126009
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053101, 0.0053222
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0206010, 0.0206480
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040077, 0.0040169
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052274, 0.0052155
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006668, 0.0006653
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036035, 0.0036117
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0180810, 0.0180398

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 218

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0132907, upper bound: 0.0133422
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0130581, upper bound: 0.0134605
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009603, 0.0009588
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0053086, 0.0053170
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0118788, 0.0118601
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0049979, 0.0050057
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0193898, 0.0194203
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0037721, 0.0037780
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0049165, 0.0049088
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006272, 0.0006262
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0033916, 0.0033969
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0170059, 0.0169792

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120230, upper bound: 0.0120181
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0120230, upper bound: 0.0120181
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009681, 0.0009506
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0052637, 0.0053606
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0119761, 0.0117597
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0049556, 0.0050468
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0192256, 0.0195796
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0037402, 0.0038090
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0049569, 0.0048673
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006323, 0.0006209
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0033629, 0.0034248
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0171453, 0.0168354

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0132955, upper bound: 0.0135397
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0132756, upper bound: 0.0135695
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010218, 0.0009692
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0053666, 0.0056576
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0126396, 0.0119896
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0050524, 0.0053264
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0196015, 0.0206643
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0038133, 0.0040200
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052315, 0.0049624
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006673, 0.0006330
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0034286, 0.0036145
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0180952, 0.0171646

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0125053, upper bound: 0.0135347
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0124888, upper bound: 0.0135693
time: 1.32 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.24 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0119703, upper bound: 0.0116331
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0119703, upper bound: 0.0116331
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0099907, upper bound: 0.0100028
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0099907, upper bound: 0.0100028
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0122038, upper bound: 0.0121882
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0122038, upper bound: 0.0121882
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0120408, upper bound: 0.0123677
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0120408, upper bound: 0.0123677
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0143288, upper bound: 0.0130399
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0142830, upper bound: 0.0130529
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0132907, upper bound: 0.0133422
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0130581, upper bound: 0.0134605
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0120230, upper bound: 0.0120181
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0120230, upper bound: 0.0120181
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0132955, upper bound: 0.0135397
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0132756, upper bound: 0.0135695
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0125053, upper bound: 0.0135347
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.24
Output dim: 4, lower bound: -0.0124888, upper bound: 0.0135693

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009818, 0.0010313
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0057102, 0.0054361
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0121447, 0.0127572
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053759, 0.0051178
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0208565, 0.0198552
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040574, 0.0038626
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0050266, 0.0052801
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006412, 0.0006735
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036481, 0.0034730
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0173867, 0.0182635

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0138239, upper bound: 0.0123821
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0136847, upper bound: 0.0125878
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009823, 0.0010307
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0057071, 0.0054390
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0121514, 0.0127504
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0053731, 0.0051206
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0208454, 0.0198660
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0040553, 0.0038647
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0050294, 0.0052773
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006415, 0.0006732
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0036462, 0.0034749
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0173962, 0.0182538

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0118199, upper bound: 0.0113731
time: 0.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0118199, upper bound: 0.0113731
time: 0.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010210, 0.0010064
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0055722, 0.0056531
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0126297, 0.0124488
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0052460, 0.0053222
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0203523, 0.0206480
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0039593, 0.0040169
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052274, 0.0051525
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006668, 0.0006573
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0035600, 0.0036117
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0180810, 0.0178220

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0126078, upper bound: 0.0128209
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0124429, upper bound: 0.0129642
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009642, 0.0009473
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0052451, 0.0053390
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0119278, 0.0117181
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0049380, 0.0050264
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0191576, 0.0195006
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0037269, 0.0037936
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0049369, 0.0048500
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006297, 0.0006187
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0033510, 0.0034110
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0170762, 0.0167759

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122835, upper bound: 0.0127215
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0125310, upper bound: 0.0126043
time: 1.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009650, 0.0009467
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0052421, 0.0053434
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0119377, 0.0117113
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0049352, 0.0050306
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0191467, 0.0195167
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0037248, 0.0037968
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0049409, 0.0048473
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006303, 0.0006183
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0033491, 0.0034138
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0170903, 0.0167662

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0129614, upper bound: 0.0133626
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0130486, upper bound: 0.0131755
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010164, 0.0009644
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0053397, 0.0056278
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0125730, 0.0119294
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0050271, 0.0052983
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0195031, 0.0205554
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0037941, 0.0039988
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052039, 0.0049375
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006638, 0.0006298
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0034114, 0.0035955
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0179999, 0.0170784

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122021, upper bound: 0.0133227
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0122805, upper bound: 0.0131503
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0010170, 0.0009639
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0053370, 0.0056310
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0125802, 0.0119235
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0050246, 0.0053013
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0194934, 0.0205671
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0037923, 0.0040011
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0052069, 0.0049351
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006642, 0.0006295
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0034097, 0.0035975
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0180101, 0.0170699

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0119880, upper bound: 0.0129129
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0118598, upper bound: 0.0130147
time: 1.22 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.08 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 4, lower bound: -0.0138239, upper bound: 0.0123821
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.08
Output dim: 4, lower bound: -0.0136847, upper bound: 0.0125878
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 4, lower bound: -0.0118199, upper bound: 0.0113731
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 4, lower bound: -0.0118199, upper bound: 0.0113731
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 4, lower bound: -0.0126078, upper bound: 0.0128209
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 4, lower bound: -0.0124429, upper bound: 0.0129642
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 4, lower bound: -0.0122835, upper bound: 0.0127215
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 4, lower bound: -0.0125310, upper bound: 0.0126043
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 4, lower bound: -0.0129614, upper bound: 0.0133626
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 4, lower bound: -0.0130486, upper bound: 0.0131755
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 4, lower bound: -0.0122021, upper bound: 0.0133227
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 4, lower bound: -0.0122805, upper bound: 0.0131503
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 4, lower bound: -0.0119880, upper bound: 0.0129129
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.08
Output dim: 4, lower bound: -0.0118598, upper bound: 0.0130147

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009248, 0.0009803
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0054280, 0.0051208
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0114405, 0.0121267
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0051102, 0.0048211
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0198257, 0.0187039
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0038569, 0.0036387
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0047352, 0.0050192
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006040, 0.0006402
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0034678, 0.0032716
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0163785, 0.0173609

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 77

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0134165, upper bound: 0.0122150
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0134136, upper bound: 0.0122150
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009330, 0.0009739
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0053924, 0.0051660
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0115415, 0.0120473
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0050768, 0.0048636
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0196959, 0.0188690
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0038316, 0.0036708
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0047770, 0.0049863
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006093, 0.0006361
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0034451, 0.0033005
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0165231, 0.0172472

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0135154, upper bound: 0.0125380
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0136227, upper bound: 0.0124429
time: 1.29 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 3.29 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.29
Output dim: 4, lower bound: -0.0134165, upper bound: 0.0122150
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.29
Output dim: 4, lower bound: -0.0134136, upper bound: 0.0122150
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 3.29
Output dim: 4, lower bound: -0.0135154, upper bound: 0.0125380
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 3.29
Output dim: 4, lower bound: -0.0136227, upper bound: 0.0124429

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009232, 0.0009798
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0054252, 0.0051116
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0114198, 0.0121204
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0051076, 0.0048123
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0198154, 0.0186700
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0038549, 0.0036321
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0047266, 0.0050166
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006029, 0.0006399
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0034660, 0.0032657
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0163489, 0.0173519

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 218

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0132078, upper bound: 0.0119589
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0130916, upper bound: 0.0120768
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009248, 0.0009786
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0054186, 0.0051208
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0114405, 0.0121057
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0051014, 0.0048211
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0197913, 0.0187039
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0038502, 0.0036387
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0047352, 0.0050105
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006040, 0.0006391
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0034618, 0.0032716
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0163785, 0.0173308

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0121089, upper bound: 0.0112670
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0121089, upper bound: 0.0112670
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009232, 0.0009597
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0053137, 0.0051119
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0114205, 0.0118714
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0050026, 0.0048126
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0194083, 0.0186711
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0037757, 0.0036323
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0047269, 0.0049135
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0006030, 0.0006268
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0033948, 0.0032659
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0163498, 0.0169954

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0110678, upper bound: 0.0108501
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0110678, upper bound: 0.0108501
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0044193, -0.0033169, -0.0044193, -0.0033169, -0.0009186, 0.0009647
1: -0.0018863, 0.0042175, -0.0018863, 0.0042175, -0.0053414, 0.0050863
2: 0.0055439, 0.0191803, 0.0055439, 0.0191803, -0.0113633, 0.0119332
3: -0.0007483, 0.0049982, -0.0007483, 0.0049982, -0.0050287, 0.0047885
4: 0.9938473, 1.0161412, 0.9938473, 1.0161412, -0.0195093, 0.0185777
5: 0.0010445, 0.0053815, 0.0010445, 0.0053815, -0.0037953, 0.0036141
6: -0.0127462, -0.0071022, -0.0127462, -0.0071022, -0.0047032, 0.0049391
7: -0.0104293, -0.0097093, -0.0104293, -0.0097093, -0.0005999, 0.0006300
8: -0.0059471, -0.0020475, -0.0059471, -0.0020475, -0.0034125, 0.0032495
9: -0.0079206, 0.0116016, -0.0079206, 0.0116016, -0.0162680, 0.0170838

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0126880, upper bound: 0.0117197
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0128213, upper bound: 0.0114738
time: 1.29 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 3.31 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.31
Output dim: 4, lower bound: -0.0132078, upper bound: 0.0119589
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.31
Output dim: 4, lower bound: -0.0130916, upper bound: 0.0120768
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.31
Output dim: 4, lower bound: -0.0121089, upper bound: 0.0112670
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.31
Output dim: 4, lower bound: -0.0121089, upper bound: 0.0112670
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.31
Output dim: 4, lower bound: -0.0110678, upper bound: 0.0108501
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.31
Output dim: 4, lower bound: -0.0110678, upper bound: 0.0108501
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.31
Output dim: 4, lower bound: -0.0126880, upper bound: 0.0117197
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.31
Output dim: 4, lower bound: -0.0128213, upper bound: 0.0114738

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.12 + 203.02 = 206.14 seconds
