## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 47.0393777385


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.8739548, 30.5588493, -8.8739548, 30.5588493, -39.4328003, 39.4328003)
1: (-10.2695408, 35.3635216, -10.2695408, 35.3635216, -45.6330643, 45.6330643)
2: (-10.8957434, 34.6628456, -10.8957434, 34.6628456, -45.5585861, 45.5585861)
3: (-15.7225132, 37.0795212, -15.7225132, 37.0795212, -52.8020287, 52.8020287)
4: (-17.3079681, 33.5112419, -17.3079681, 33.5112419, -50.8191986, 50.8192024)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.53 + 1.50 = 3.03 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -47.1809205, upper bound: 47.1809205

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1760167, upper bound: 47.1686500
time: 0.81 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.80 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.75 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 4, lower bound: -47.1760167, upper bound: 47.1686500
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.3289332, 15.6493454, -7.7389174, 26.9503441, -31.2792778, 23.3882599
1: -4.9227071, 18.1375618, -8.9462147, 31.2036190, -36.1263275, 27.0837765
2: -5.4355350, 17.6055393, -9.5667200, 30.5406857, -35.9762154, 27.1722584
3: -7.7107296, 19.0719624, -13.7645473, 32.7263031, -40.4370308, 32.8365059
4: -9.0420084, 16.7446327, -15.3124866, 29.4453449, -38.4873466, 32.0571213

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.49 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.68 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -13.8669882, 47.8264809, -8.7006226, 29.9893379, -43.8563232, 56.4315414
1: -16.4424095, 55.6123199, -10.0655127, 34.7013779, -51.1437759, 65.5285339
2: -17.0157623, 54.3669052, -10.6880798, 34.0129738, -51.0287285, 64.9423981
3: -25.0840302, 58.4735527, -15.4150457, 36.3897820, -61.4738121, 73.7443390
4: -27.2452755, 52.4822922, -16.9911537, 32.8825493, -60.1278229, 69.4734497

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.50 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
time: 0.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.56 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 4, lower bound: -47.1679742, upper bound: 47.1679742

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -4.3289332, 15.6493454, -4.3289332, 15.6493454, -19.9782753, 19.9782753
1: -4.9227071, 18.1375618, -4.9227071, 18.1375618, -23.0602684, 23.0602684
2: -5.4355350, 17.6055393, -5.4355350, 17.6055393, -23.0410748, 23.0410748
3: -7.7107296, 19.0719624, -7.7107296, 19.0719624, -26.7826920, 26.7826920
4: -9.0420084, 16.7446327, -9.0420084, 16.7446327, -25.7866364, 25.7866364

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1760005, upper bound: 47.1680529
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1760167, upper bound: 47.1686500
time: 0.46 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -4.3289332, 15.6493454, -13.0472050, 44.9861183, -49.1897354, 28.6965466
1: -4.9227071, 18.1375618, -15.3804827, 52.3511429, -57.1140938, 33.5180435
2: -5.4355350, 17.6055393, -16.0566273, 51.1194839, -56.4232712, 33.6621666
3: -7.7107296, 19.0719624, -23.5816936, 55.1130180, -62.6925659, 42.6536484
4: -9.0420084, 16.7446327, -25.8167171, 49.3024445, -58.3444481, 42.5613441

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1127253, upper bound: 47.0679021
time: 0.42 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0970870, upper bound: 47.0655948
time: 0.84 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -13.8669882, 47.8264809, -4.3289332, 15.6493454, -29.5163288, 52.0300331
1: -16.4424095, 55.6123199, -4.9227071, 18.1375618, -34.5799713, 60.3761864
2: -17.0157623, 54.3669052, -5.4355350, 17.6055393, -34.6212921, 59.6714478
3: -25.0840302, 58.4735527, -7.7107296, 19.0719624, -44.1559906, 66.0495911
4: -27.2452755, 52.4822922, -9.0420084, 16.7446327, -43.9899063, 61.5242958

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0660411, upper bound: 47.0913122
time: 0.44 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.46 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -13.8669882, 47.8264809, -13.0686092, 45.0560722, -58.8123169, 60.7855453
1: -16.4424095, 55.6123199, -15.4047651, 52.4332733, -68.6767883, 70.8160324
2: -17.0157623, 54.3669052, -16.0822449, 51.1983566, -68.0413132, 70.2737198
3: -25.0840302, 58.4735527, -23.6169319, 55.1978226, -80.0572128, 81.8583908
4: -27.2452755, 52.4822922, -25.8539028, 49.3803787, -76.5030136, 78.2091293

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0913122, upper bound: 47.0660411
time: 0.48 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.86 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 4, lower bound: -47.1760005, upper bound: 47.1680529
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 4, lower bound: -47.1760167, upper bound: 47.1686500
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 4, lower bound: -47.1127253, upper bound: 47.0679021
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 4, lower bound: -47.0970870, upper bound: 47.0655948
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 4, lower bound: -47.0660411, upper bound: 47.0913122
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 4, lower bound: -47.0913122, upper bound: 47.0660411
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.86
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.5543404, 13.2244720, -4.3289332, 15.6493454, -19.2036800, 17.5534039
1: -4.0038738, 15.3502703, -4.9227071, 18.1375618, -22.1414356, 20.2729778
2: -4.5091252, 14.8414526, -5.4355350, 17.6055393, -22.1146641, 20.2769871
3: -6.3346443, 16.1530743, -7.7107296, 19.0719624, -25.4066048, 23.8638039
4: -7.6518283, 14.0251284, -9.0420084, 16.7446327, -24.3964615, 23.0671349

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1760295, upper bound: 47.1718362
time: 0.46 seconds

## Relational analysis of NS_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1780919, upper bound: 47.1751049
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1746890, upper bound: 47.1741960
time: 0.41 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -4.1999273, 15.2384434, -4.3289332, 15.6493454, -19.8492699, 19.5673771
1: -4.7570848, 17.6634674, -4.9227071, 18.1375618, -22.8946457, 22.5861740
2: -5.2808690, 17.1281166, -5.4355350, 17.6055393, -22.8864079, 22.5636520
3: -7.4648337, 18.5741806, -7.7107296, 19.0719624, -26.5367947, 26.2849102
4: -8.8112183, 16.2702618, -9.0420084, 16.7446327, -25.5558510, 25.3122635

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1760523, upper bound: 47.1722428
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712896, upper bound: 47.1712896
time: 0.44 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.7740495, 13.7595510, -13.0472050, 44.9861183, -48.6287804, 26.8067551
1: -4.2787447, 15.9366035, -15.3804827, 52.3511429, -56.4661217, 31.3170834
2: -4.7472148, 15.4557161, -16.0566273, 51.1194839, -55.7243843, 31.5123405
3: -6.7212100, 16.7551899, -23.5816936, 55.1130180, -61.6897163, 40.3368835
4: -7.9449539, 14.6762562, -25.8167171, 49.3024445, -57.2312584, 40.4929695

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1127253, upper bound: 47.0674529
time: 0.46 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1127253, upper bound: 47.0679021
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.0988472, 11.2521486, -12.6583986, 43.6222992, -46.6068840, 23.9105453
1: -3.5240169, 12.9181213, -14.9161911, 50.7526703, -54.1298103, 27.8343105
2: -3.8821015, 12.6087656, -15.5776339, 49.5574989, -53.3354149, 28.1863976
3: -5.4794450, 13.5591316, -22.8663597, 53.4346733, -58.8135300, 36.4254723
4: -6.4332042, 12.0126858, -25.0497627, 47.7958755, -54.2290802, 37.0624466

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0969853, upper bound: 47.0653220
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0970870, upper bound: 47.0655948
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -13.8669882, 47.8264809, -3.7740495, 13.7595510, -27.6265392, 51.4690704
1: -16.4424095, 55.6123199, -4.2787447, 15.9366035, -32.3790092, 59.7282104
2: -17.0157623, 54.3669052, -4.7472148, 15.4557161, -32.4714737, 58.9725571
3: -25.0840302, 58.4735527, -6.7212100, 16.7551899, -41.8392181, 65.0467453
4: -27.2452755, 52.4822922, -7.9449539, 14.6762562, -41.9215317, 60.4102173

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0674529, upper bound: 47.1127253
time: 0.95 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0679021, upper bound: 47.1127253
time: 0.46 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -13.3720465, 46.1078644, -3.0988472, 11.2521486, -24.6241894, 49.0915413
1: -15.8380375, 53.6062202, -3.5240169, 12.9181213, -28.7561588, 56.9821205
2: -16.4121265, 52.3989677, -3.8821015, 12.6087656, -29.0208931, 56.1765633
3: -24.1741505, 56.3749924, -5.4794450, 13.5591316, -37.7332840, 61.7490196
4: -26.2971077, 50.5764999, -6.4332042, 12.0126858, -38.3097878, 57.0097046

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0653220, upper bound: 47.0969853
time: 0.49 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0655948, upper bound: 47.0970870
time: 0.45 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -12.9033422, 44.5502892, -13.0686092, 45.0560722, -57.8463936, 57.4393959
1: -15.2967863, 51.7916565, -15.4047651, 52.4332733, -67.5292511, 66.9043961
2: -15.8526278, 50.6367798, -16.0822449, 51.1983566, -66.8677444, 66.4635925
3: -23.3680229, 54.4814682, -23.6169319, 55.1978226, -78.3276443, 77.7797012
4: -25.4215183, 48.8891449, -25.8539028, 49.3803787, -74.6479645, 74.5607452

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0815670, upper bound: 47.0605102
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0815670, upper bound: 47.0660411
time: 0.48 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -11.2247467, 38.5842094, -12.6741180, 43.6722984, -54.7946320, 51.2583199
1: -13.2530212, 44.7912140, -14.9336834, 50.8111649, -63.8974571, 59.6948700
2: -13.7886238, 43.8066788, -15.5960121, 49.6139755, -63.2500496, 59.3722725
3: -20.2533207, 47.1470337, -22.8917484, 53.4951210, -73.5625610, 69.9623947
4: -22.0918694, 42.3267174, -25.0763969, 47.8517303, -69.8648224, 67.3833694

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0393201
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.19 seconds
NS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 4, lower bound: -47.1780919, upper bound: 47.1751049
NS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 4, lower bound: -47.1746890, upper bound: 47.1741960
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 4, lower bound: -47.1760523, upper bound: 47.1722428
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 4, lower bound: -47.1712896, upper bound: 47.1712896
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 4, lower bound: -47.1127253, upper bound: 47.0674529
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 4, lower bound: -47.1127253, upper bound: 47.0679021
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 4, lower bound: -47.0969853, upper bound: 47.0653220
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 4, lower bound: -47.0970870, upper bound: 47.0655948
NS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 4, lower bound: -47.0674529, upper bound: 47.1127253
NS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 4, lower bound: -47.0679021, upper bound: 47.1127253
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 4, lower bound: -47.0653220, upper bound: 47.0969853
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 4, lower bound: -47.0655948, upper bound: 47.0970870
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 4, lower bound: -47.0815670, upper bound: 47.0605102
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 4, lower bound: -47.0815670, upper bound: 47.0660411
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0393201
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536

## BFS NS instance: NS_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -1.9142306, 7.8301115, -4.2743053, 15.4642897, -17.3785210, 12.1044168
1: -2.1368051, 9.0999527, -4.8586745, 17.9214115, -20.0582161, 13.9586277
2: -2.4833076, 8.7128201, -5.3677964, 17.3939915, -19.8772984, 14.0806160
3: -3.4692547, 9.6202745, -7.6127563, 18.8463631, -22.3156166, 17.2330303
4: -4.5356984, 8.0148392, -8.9366360, 16.5383377, -21.0740337, 16.9514732

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1736167, upper bound: 47.1650261
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1736167, upper bound: 47.1745560
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -3.1026962, 11.7314358, -4.3289332, 15.6493454, -18.7520370, 16.0603676
1: -3.4796655, 13.6191635, -4.9227071, 18.1375618, -21.6172276, 18.5418701
2: -3.9527643, 13.1387415, -5.4355350, 17.6055393, -21.5583019, 18.5742760
3: -5.5099001, 14.3484335, -7.7107296, 19.0719624, -24.5818634, 22.0591621
4: -6.7972994, 12.3457012, -9.0420084, 16.7446327, -23.5419312, 21.3877029

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1734237, upper bound: 47.1647822
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1741167, upper bound: 47.1736057
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -3.9656982, 14.4664125, -4.3289332, 15.6493454, -19.6150398, 18.7953434
1: -4.4834776, 16.7684517, -4.9227071, 18.1375618, -22.6210384, 21.6911583
2: -4.9920673, 16.2456532, -5.4355350, 17.6055393, -22.5976067, 21.6811886
3: -7.0487165, 17.6414280, -7.7107296, 19.0719624, -26.1206779, 25.3521576
4: -8.3776455, 15.4009609, -9.0420084, 16.7446327, -25.1222763, 24.4429646

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712896, upper bound: 47.1712896
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712896, upper bound: 47.1712896
time: 0.47 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -6.0737481, 20.9647923, -4.3120708, 15.5935297, -21.6672726, 25.2768631
1: -6.9062757, 24.3164692, -4.9014654, 18.0736427, -24.9799156, 29.2179337
2: -7.5086946, 23.6424351, -5.4154849, 17.5414486, -25.0501442, 29.0579185
3: -10.6806793, 25.5491333, -7.6792197, 19.0058842, -29.6865616, 33.2283516
4: -12.1793528, 22.6682663, -9.0119915, 16.6813202, -28.8606720, 31.6802559

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712896, upper bound: 47.1712896
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712896, upper bound: 47.1712896
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -3.0475309, 11.4952679, -13.0472050, 44.9861183, -47.8932037, 24.5424728
1: -3.4280396, 13.3304625, -15.3804827, 52.3511429, -55.6096954, 28.7109451
2: -3.8826449, 12.8824883, -16.0566273, 51.1194839, -54.8574715, 28.9391098
3: -5.4488935, 14.0269852, -23.5816936, 55.1130180, -60.4158745, 37.6086807
4: -6.6468649, 12.1436949, -25.8167171, 49.3024445, -55.9251099, 37.9604034

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1127253, upper bound: 47.0674529
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1127253, upper bound: 47.0674529
time: 0.47 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -3.6656094, 13.4249163, -13.0472050, 44.9861183, -48.5208893, 26.4721203
1: -4.1421041, 15.5496635, -15.3804827, 52.3511429, -56.3290634, 30.9301453
2: -4.6192713, 15.0691986, -16.0566273, 51.1194839, -55.5963898, 31.1258221
3: -6.5204768, 16.3473377, -23.5816936, 55.1130180, -61.4855003, 39.9290314
4: -7.7538829, 14.2926893, -25.8167171, 49.3024445, -57.0327110, 40.1094017

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1127253, upper bound: 47.0679021
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1127253, upper bound: 47.0679021
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -2.4840326, 9.2791185, -12.6583986, 43.6222992, -45.9861717, 21.9375172
1: -2.8140178, 10.6496401, -14.9161911, 50.7526703, -53.4169197, 25.5658302
2: -3.1398907, 10.3862915, -15.5776339, 49.5574989, -52.5910950, 25.9639244
3: -4.4031024, 11.2005587, -22.8663597, 53.4346733, -57.7360458, 34.0669136
4: -5.3245440, 9.8329000, -25.0497627, 47.7958755, -53.1204185, 34.8826637

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0924406, upper bound: 47.0593205
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0963289, upper bound: 47.0651247
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0963289, upper bound: 47.0653220
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -3.0158601, 11.0064144, -12.6583986, 43.6222992, -46.5242653, 23.6648140
1: -3.4244437, 12.6325045, -14.9161911, 50.7526703, -54.0301094, 27.5486946
2: -3.7862558, 12.3272820, -15.5776339, 49.5574989, -53.2386551, 27.9049091
3: -5.3343902, 13.2565231, -22.8663597, 53.4346733, -58.6649208, 36.1228714
4: -6.2886052, 11.7358017, -25.0497627, 47.7958755, -54.0844803, 36.7855644

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0925437, upper bound: 47.0597725
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0964260, upper bound: 47.0653976
time: 0.49 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0964260, upper bound: 47.0655948
time: 0.45 seconds

## BFS NS instance: NS_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -13.8669882, 47.8264809, -3.0475309, 11.4952679, -25.3622532, 50.7334976
1: -16.4424095, 55.6123199, -3.4280396, 13.3304625, -29.7728729, 58.8717880
2: -17.0157623, 54.3669052, -3.8826449, 12.8824883, -29.8982487, 58.1056480
3: -25.0840302, 58.4735527, -5.4488935, 14.0269852, -39.1110153, 63.7728844
4: -27.2452755, 52.4822922, -6.6468649, 12.1436949, -39.3889694, 59.1040688

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0674529, upper bound: 47.1127253
time: 0.47 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0674529, upper bound: 47.1127253
time: 0.50 seconds

## BFS NS instance: NS_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -13.8669882, 47.8264809, -3.6656094, 13.4249163, -27.2919025, 51.3611832
1: -16.4424095, 55.6123199, -4.1421041, 15.5496635, -31.9920692, 59.5911560
2: -17.0157623, 54.3669052, -4.6192713, 15.0691986, -32.0849571, 58.8445663
3: -25.0840302, 58.4735527, -6.5204768, 16.3473377, -41.4313660, 64.8425064
4: -27.2452755, 52.4822922, -7.7538829, 14.2926893, -41.5379639, 60.2116661

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0679021, upper bound: 47.1127253
time: 0.77 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0679021, upper bound: 47.1127253
time: 0.47 seconds

## BFS NS instance: NS_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -13.3720465, 46.1078644, -2.4840326, 9.2791185, -22.6511612, 48.4708290
1: -15.8380375, 53.6062202, -2.8140178, 10.6496401, -26.4876785, 56.2692299
2: -16.4121265, 52.3989677, -3.1398907, 10.3862915, -26.7984180, 55.4322433
3: -24.1741505, 56.3749924, -4.4031024, 11.2005587, -35.3747101, 60.6715317
4: -26.2971077, 50.5764999, -5.3245440, 9.8329000, -36.1300087, 55.9010429

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0963289
time: 0.47 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0969853
time: 0.54 seconds

## BFS NS instance: NS_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -13.3720465, 46.1078644, -3.0158601, 11.0064144, -24.3784542, 49.0089226
1: -15.8380375, 53.6062202, -3.4244437, 12.6325045, -28.4705429, 56.8824196
2: -16.4121265, 52.3989677, -3.7862558, 12.3272820, -28.7394009, 56.0798035
3: -24.1741505, 56.3749924, -5.3343902, 13.2565231, -37.4306717, 61.6004105
4: -26.2971077, 50.5764999, -6.2886052, 11.7358017, -38.0329056, 56.8651047

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0597725, upper bound: 47.0925437
time: 0.76 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0653976, upper bound: 47.0964260
time: 0.55 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0653976, upper bound: 47.0970870
time: 0.48 seconds

## BFS NS instance: NS_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -12.5650520, 43.4993477, -13.0686092, 45.0560722, -57.5117188, 56.3821754
1: -14.8664980, 50.5900497, -15.4047651, 52.4332733, -67.0946732, 65.7003021
2: -15.4648190, 49.4217529, -16.0822449, 51.1983566, -66.4829636, 65.2443390
3: -22.7446213, 53.2128296, -23.6169319, 55.1978226, -77.7038498, 76.5078506
4: -24.8334503, 47.7021942, -25.8539028, 49.3803787, -74.0522766, 73.3689728

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0815670, upper bound: 47.0605102
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0815670, upper bound: 47.0605102
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -12.6311121, 43.6649780, -13.0686092, 45.0560722, -57.5747337, 56.5489655
1: -14.9605846, 50.7695885, -15.4047651, 52.4332733, -67.1938095, 65.8795700
2: -15.5325613, 49.6204185, -16.0822449, 51.1983566, -66.5463562, 65.4429550
3: -22.8752594, 53.4159927, -23.6169319, 55.1978226, -77.8334045, 76.7087173
4: -24.9373646, 47.8990097, -25.8539028, 49.3803787, -74.1552734, 73.5663147

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0815670, upper bound: 47.0660411
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0815670, upper bound: 47.0660411
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -10.7382069, 37.0713997, -12.6741180, 43.6722984, -54.3055611, 49.7455101
1: -12.6549892, 43.0521164, -14.9336834, 50.8111649, -63.2951698, 57.9477921
2: -13.2225256, 42.0720711, -15.5960121, 49.6139755, -62.6830330, 57.6328697
3: -19.3875694, 45.3159103, -22.8917484, 53.4951210, -72.6956863, 68.1233063
4: -21.2321262, 40.6393242, -25.0763969, 47.8517303, -68.9975586, 65.6926498

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0393201
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0393201
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -10.9528446, 37.6870155, -12.6741180, 43.6722984, -54.5235596, 50.3611259
1: -12.9132843, 43.7553520, -14.9336834, 50.8111649, -63.5580864, 58.6548615
2: -13.4681273, 42.7748718, -15.5960121, 49.6139755, -62.9278908, 58.3347816
3: -19.7512875, 46.0691261, -22.8917484, 53.4951210, -73.0593872, 68.8779449
4: -21.6068287, 41.3166008, -25.0763969, 47.8517303, -69.3717194, 66.3691940

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
time: 0.91 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.06 seconds
NS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.1736167, upper bound: 47.1650261
NS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.1736167, upper bound: 47.1745560
NS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.1734237, upper bound: 47.1647822
NS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.1741167, upper bound: 47.1736057
NS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.1712896, upper bound: 47.1712896
NS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.1712896, upper bound: 47.1712896
NS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.1712896, upper bound: 47.1712896
NS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.1712896, upper bound: 47.1712896
NS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.1127253, upper bound: 47.0674529
NS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.1127253, upper bound: 47.0674529
NS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.1127253, upper bound: 47.0679021
NS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.1127253, upper bound: 47.0679021
NS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0963289, upper bound: 47.0651247
NS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0963289, upper bound: 47.0653220
NS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0964260, upper bound: 47.0653976
NS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0964260, upper bound: 47.0655948
NS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0674529, upper bound: 47.1127253
NS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0674529, upper bound: 47.1127253
NS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0679021, upper bound: 47.1127253
NS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0679021, upper bound: 47.1127253
NS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0963289
NS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0969853
NS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0653976, upper bound: 47.0964260
NS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0653976, upper bound: 47.0970870
NS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0815670, upper bound: 47.0605102
NS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0815670, upper bound: 47.0605102
NS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0815670, upper bound: 47.0660411
NS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0815670, upper bound: 47.0660411
NS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0393201
NS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0401747, upper bound: 47.0393201
NS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536
NS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.06
Output dim: 4, lower bound: -47.0404536, upper bound: 47.0404536

## BFS NS instance: NS_A1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -1.7854135, 7.3783250, -3.4587708, 12.8446541, -14.6300678, 10.8370953
1: -1.9944807, 8.5729265, -3.9751954, 14.8369598, -16.8314400, 12.5481215
2: -2.3179524, 8.2031059, -4.3432884, 14.4518232, -16.7697754, 12.5463934
3: -3.2469039, 9.0642776, -6.2488904, 15.5703440, -18.8172455, 15.3131666
4: -4.2707787, 7.5267539, -7.2941327, 13.7042961, -17.9750710, 14.8208866

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1656192, upper bound: 47.1628866
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1714059, upper bound: 47.1630303
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -1.9142306, 7.8301115, -4.1416893, 14.9934130, -16.9076405, 11.9718008
1: -2.1368051, 9.0999527, -4.7075543, 17.3641205, -19.5009251, 13.8075066
2: -2.4833076, 8.7128201, -5.2026291, 16.8603134, -19.3436184, 13.9154491
3: -3.4692547, 9.6202745, -7.3769655, 18.2618446, -21.7310982, 16.9972382
4: -4.5356984, 8.0148392, -8.6624832, 16.0341702, -20.5698643, 16.6773148

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1656192, upper bound: 47.1715351
time: 0.46 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1714059, upper bound: 47.1711616
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -2.8822157, 11.0232143, -3.5011101, 12.9874105, -15.8696251, 14.5243244
1: -3.2299409, 12.7959824, -4.0240164, 15.0041294, -18.2340698, 16.8199978
2: -3.6806459, 12.3376141, -4.3955851, 14.6139641, -18.2946091, 16.7331982
3: -5.1305866, 13.4789286, -6.3240385, 15.7447224, -20.8753071, 19.8029671
4: -6.3740869, 11.5627041, -7.3757968, 13.8620195, -20.2361050, 18.9385014

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1719834, upper bound: 47.1599844
time: 0.43 seconds

## Relational analysis of NS_A1_B1_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1713122, upper bound: 47.1629633
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3.1026962, 11.7314358, -4.1940603, 15.1722717, -18.2749672, 15.9254961
1: -3.4796655, 13.6191635, -4.7692404, 17.5731640, -21.0528297, 18.3884029
2: -3.9527643, 13.1387415, -5.2681537, 17.0648670, -21.0176296, 18.4068947
3: -5.5099001, 14.3484335, -7.4714885, 18.4803181, -23.9902191, 21.8199215
4: -6.7972994, 12.3457012, -8.7648716, 16.2336521, -23.0309525, 21.1105728

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1422733, upper bound: 47.1634553
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1722675, upper bound: 47.1706185
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -3.9656982, 14.4664125, -4.0885248, 14.8552752, -18.8209724, 18.5549355
1: -4.4834776, 16.7684517, -4.6401472, 17.2174129, -21.7008877, 21.4085999
2: -4.9920673, 16.2456532, -5.1397719, 16.6970844, -21.6891499, 21.3854218
3: -7.0487165, 17.6414280, -7.2818685, 18.1143341, -25.1630516, 24.9232960
4: -8.3776455, 15.4009609, -8.5971518, 15.8510361, -24.2286816, 23.9981117

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1712657, upper bound: 47.1705583
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1725106, upper bound: 47.1698398
time: 0.50 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -3.9656982, 14.4664125, -6.2036738, 21.3862705, -25.3519669, 20.6700859
1: -4.4834776, 16.7684517, -7.0725803, 24.8031158, -29.2865906, 23.8410301
2: -4.9920673, 16.2456532, -7.6664505, 24.1312027, -29.1232681, 23.9121037
3: -7.0487165, 17.6414280, -10.9312239, 26.0610123, -33.1097298, 28.5726509
4: -8.3776455, 15.4009609, -12.4161844, 23.1557178, -31.5333633, 27.8171463

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1733865, upper bound: 47.1690427
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1733629, upper bound: 47.1688688
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -6.0737481, 20.9647923, -4.0885248, 14.8552752, -20.9290218, 25.0533180
1: -6.9062757, 24.3164692, -4.6401472, 17.2174129, -24.1236877, 28.9566154
2: -7.5086946, 23.6424351, -5.1397719, 16.6970844, -24.2057800, 28.7822056
3: -10.6806793, 25.5491333, -7.2818685, 18.1143341, -28.7950115, 32.8310013
4: -12.1793528, 22.6682663, -8.5971518, 15.8510361, -28.0303879, 31.2654190

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1695583, upper bound: 47.1655877
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1689582, upper bound: 47.1689582
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -6.0737481, 20.9647923, -6.2036738, 21.3862705, -27.4600182, 27.1684666
1: -6.9062757, 24.3164692, -7.0725803, 24.8031158, -31.7093925, 31.3890457
2: -7.5086946, 23.6424351, -7.6664505, 24.1312027, -31.6398964, 31.3088856
3: -10.6806793, 25.5491333, -10.9312239, 26.0610123, -36.7416916, 36.4803581
4: -12.1793528, 22.6682663, -12.4161844, 23.1557178, -35.3350716, 35.0844460

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678896, upper bound: 47.1672706
time: 0.46 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1678676, upper bound: 47.1678676
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -3.0475309, 11.4952679, -12.3234282, 42.5332031, -45.3697395, 23.8186951
1: -3.4280396, 13.3304625, -14.5424137, 49.4741707, -52.6415176, 27.8728752
2: -3.8826449, 12.8824883, -15.1788206, 48.3280334, -51.9860687, 28.0613098
3: -5.4488935, 14.0269852, -22.2992535, 52.0985527, -57.3133507, 36.3262367
4: -6.6468649, 12.1436949, -24.4093018, 46.6316299, -53.1985970, 36.5529938

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1084331, upper bound: 47.0618519
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1118286, upper bound: 47.0672008
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A1_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1118286, upper bound: 47.0674529
time: 0.48 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -3.0475309, 11.4952679, -10.9450626, 37.6084023, -40.6559334, 22.4403267
1: -3.4280396, 13.3304625, -12.8934517, 43.6690178, -47.0970573, 26.2239132
2: -3.8826449, 12.8824883, -13.4669304, 42.6933708, -46.5760155, 26.3494148
3: -5.4488935, 14.0269852, -19.7413235, 45.9990692, -51.4479599, 33.7683067
4: -6.6468649, 12.1436949, -21.6055470, 41.2416763, -47.8885345, 33.7492371

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1084331, upper bound: 47.0618519
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1118286, upper bound: 47.0672009
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1118286, upper bound: 47.0674529
time: 0.46 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -3.6656094, 13.4249163, -12.3234282, 42.5332031, -45.9974289, 25.7483444
1: -4.1421041, 15.5496635, -14.5424137, 49.4741707, -53.3608856, 30.0920773
2: -4.6192713, 15.0691986, -15.1788206, 48.3280334, -52.7249870, 30.2480202
3: -6.5204768, 16.3473377, -22.2992535, 52.0985527, -58.3829803, 38.6465912
4: -7.7538829, 14.2926893, -24.4093018, 46.6316299, -54.3061943, 38.7019920

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0921409, upper bound: 47.0621750
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1084331, upper bound: 47.0622656
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_A2_B1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1118286, upper bound: 47.0676431
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A1_A2_B1_B2

### Relational analysis result of NS_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1118286, upper bound: 47.0677933
time: 0.51 seconds

## BFS NS instance: NS_A1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -3.6656094, 13.4249163, -10.9450626, 37.6084023, -41.2740097, 24.3699780
1: -4.1421041, 15.5496635, -12.8934517, 43.6690178, -47.8111229, 28.4431095
2: -4.6192713, 15.0691986, -13.4669304, 42.6933708, -47.3126411, 28.5361290
3: -6.5204768, 16.3473377, -19.7413235, 45.9990692, -52.5195389, 36.0886574
4: -7.7538829, 14.2926893, -21.6055470, 41.2416763, -48.9955559, 35.8982353

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1084331, upper bound: 47.0622656
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_A2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1118286, upper bound: 47.0676431
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1118286, upper bound: 47.0677933
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -2.4840326, 9.2791185, -12.3225765, 42.6090965, -44.9635735, 21.6016941
1: -2.8140178, 10.6496401, -14.4988194, 49.5931015, -52.2479286, 25.1484585
2: -3.1398907, 10.3862915, -15.1882353, 48.3904610, -51.4187012, 25.5745277
3: -4.4031024, 11.2005587, -22.2570763, 52.2029190, -56.4980049, 33.4576340
4: -5.3245440, 9.8329000, -24.4545937, 46.6567001, -51.9812431, 34.2874947

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0963289, upper bound: 47.0651248
time: 0.45 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0963289, upper bound: 47.0651248
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -2.4840326, 9.2791185, -12.4347200, 42.8997154, -45.2560349, 21.7138386
1: -2.8140178, 10.6496401, -14.6424026, 49.9167252, -52.5747681, 25.2920380
2: -3.1398907, 10.3862915, -15.3131428, 48.7283630, -51.7555428, 25.6994343
3: -4.4031024, 11.2005587, -22.4587460, 52.5612144, -56.8539238, 33.6593056
4: -5.3245440, 9.8329000, -24.6478767, 46.9871902, -52.3117332, 34.4807777

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0963289, upper bound: 47.0653220
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0963289, upper bound: 47.0653220
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -3.0158601, 11.0064144, -12.3225765, 42.6090965, -45.5016670, 23.3289909
1: -3.4244437, 12.6325045, -14.4988194, 49.5931015, -52.8611183, 27.1313248
2: -3.7862558, 12.3272820, -15.1882353, 48.3904610, -52.0662613, 27.5155125
3: -5.3343902, 13.2565231, -22.2570763, 52.2029190, -57.4268799, 35.5135994
4: -6.2886052, 11.7358017, -24.4545937, 46.6567001, -52.9453049, 36.1903954

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0964260, upper bound: 47.0653976
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0964260, upper bound: 47.0653976
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -3.0158601, 11.0064144, -12.4347200, 42.8997154, -45.7941284, 23.4411335
1: -3.4244437, 12.6325045, -14.6424026, 49.9167252, -53.1879578, 27.2749062
2: -3.7862558, 12.3272820, -15.3131428, 48.7283630, -52.4031029, 27.6404190
3: -5.3343902, 13.2565231, -22.4587460, 52.5612144, -57.7827988, 35.7152634
4: -6.2886052, 11.7358017, -24.6478767, 46.9871902, -53.2757950, 36.3836784

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0964260, upper bound: 47.0655185
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0964260, upper bound: 47.0655185
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -12.9033422, 44.5502892, -3.0475309, 11.4952679, -24.3986092, 47.3873444
1: -15.2967863, 51.7916565, -3.4280396, 13.3304625, -28.6272488, 54.9601440
2: -15.8526278, 50.6367798, -3.8826449, 12.8824883, -28.7351151, 54.2955170
3: -23.3680229, 54.4814682, -5.4488935, 14.0269852, -37.3950005, 59.6941910
4: -25.4215183, 48.8891449, -6.6468649, 12.1436949, -37.5652008, 55.4556961

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0618519, upper bound: 47.1084331
time: 0.75 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0672009, upper bound: 47.1118286
time: 0.50 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0672009, upper bound: 47.1127253
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -11.2247467, 38.5842094, -3.0475309, 11.4952679, -22.7200146, 41.6314201
1: -13.2530212, 44.7912140, -3.4280396, 13.3304625, -26.5834846, 48.2192497
2: -13.7886238, 43.8066788, -3.8826449, 12.8824883, -26.6711121, 47.6893234
3: -20.2533207, 47.1470337, -5.4488935, 14.0269852, -34.2803040, 52.5959282
4: -22.0918694, 42.3267174, -6.6468649, 12.1436949, -34.2355652, 48.9735832

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0618519, upper bound: 47.1084331
time: 0.53 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0672009, upper bound: 47.1118286
time: 0.52 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0672009, upper bound: 47.1127253
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -12.9033422, 44.5502892, -3.6656094, 13.4249163, -26.3282585, 48.0150299
1: -15.2967863, 51.7916565, -4.1421041, 15.5496635, -30.8464489, 55.6795120
2: -15.8526278, 50.6367798, -4.6192713, 15.0691986, -30.9218254, 55.0344353
3: -23.3680229, 54.4814682, -6.5204768, 16.3473377, -39.7153511, 60.7638168
4: -25.4215183, 48.8891449, -7.7538829, 14.2926893, -39.7142029, 56.5632935

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0621750, upper bound: 47.0921409
time: 0.85 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0622656, upper bound: 47.1084331
time: 0.85 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0672009, upper bound: 47.1118286
time: 0.89 seconds

## Relational analysis of NS_A2_B1_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0672009, upper bound: 47.1127253
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -11.2247467, 38.5842094, -3.6656094, 13.4249163, -24.6496620, 42.2498169
1: -13.2530212, 44.7912140, -4.1421041, 15.5496635, -28.8026848, 48.9333191
2: -13.7886238, 43.8066788, -4.6192713, 15.0691986, -28.8578224, 48.4259453
3: -20.2533207, 47.1470337, -6.5204768, 16.3473377, -36.6006546, 53.6675110
4: -22.0918694, 42.3267174, -7.7538829, 14.2926893, -36.3845596, 50.0806007

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0622656, upper bound: 47.1084331
time: 0.54 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0621750, upper bound: 47.0921409
time: 0.83 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0672009, upper bound: 47.1118286
time: 0.52 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0672009, upper bound: 47.1127253
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -12.9706287, 44.8518372, -2.4840326, 9.2791185, -22.2497482, 47.2048187
1: -15.3335314, 52.1678963, -2.8140178, 10.6496401, -25.9831715, 54.8215675
2: -15.9433374, 50.9541473, -3.1398907, 10.3862915, -26.3296280, 53.9807510
3: -23.4426384, 54.8571281, -4.4031024, 11.2005587, -34.6431961, 59.1480370
4: -25.5908947, 49.1645584, -5.3245440, 9.8329000, -35.4237938, 54.4891014

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_B1_A1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0963289
time: 0.88 seconds

## Relational analysis of NS_A2_B1_B2_B1_A1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0963289
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -13.0930233, 45.2032700, -2.4840326, 9.2791185, -22.3721428, 47.5594978
1: -15.4940767, 52.5616570, -2.8140178, 10.6496401, -26.1437168, 55.2196312
2: -16.0842552, 51.3607330, -3.1398907, 10.3862915, -26.4705467, 54.3880768
3: -23.6692028, 55.2853775, -4.4031024, 11.2005587, -34.8697624, 59.5750504
4: -25.8023987, 49.5632095, -5.3245440, 9.8329000, -35.6352997, 54.8877525

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_B1_A2_A1

### Relational analysis result of NS_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0969853
time: 0.52 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0969853
time: 0.53 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -12.9706287, 44.8518372, -3.0158601, 11.0064144, -23.9770432, 47.7429123
1: -15.3335314, 52.1678963, -3.4244437, 12.6325045, -27.9660358, 55.4347534
2: -15.9433374, 50.9541473, -3.7862558, 12.3272820, -28.2706146, 54.6283112
3: -23.4426384, 54.8571281, -5.3343902, 13.2565231, -36.6991577, 60.0769119
4: -25.5908947, 49.1645584, -6.2886052, 11.7358017, -37.3266945, 55.4531631

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0964260
time: 0.78 seconds

## Relational analysis of NS_A2_B1_B2_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0964260
time: 0.51 seconds

## BFS NS instance: NS_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -13.0930233, 45.2032700, -3.0158601, 11.0064144, -24.0994377, 48.0975876
1: -15.4940767, 52.5616570, -3.4244437, 12.6325045, -28.1265812, 55.8328209
2: -16.0842552, 51.3607330, -3.7862558, 12.3272820, -28.4115334, 55.0356369
3: -23.6692028, 55.2853775, -5.3343902, 13.2565231, -36.9257240, 60.5039291
4: -25.8023987, 49.5632095, -6.2886052, 11.7358017, -37.5382004, 55.8518143

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_B2_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0970870
time: 0.55 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0963289
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -12.5650520, 43.4993477, -12.3291254, 42.5519447, -54.9359093, 55.6396103
1: -14.8664980, 50.5900497, -14.5488901, 49.4961662, -64.0644150, 64.8435516
2: -15.4648190, 49.4217529, -15.1856499, 48.3492012, -63.5522308, 64.3372879
3: -22.7446213, 53.2128296, -22.3086872, 52.1212234, -74.5373993, 75.1860504
4: -24.8334503, 47.7021942, -24.4192924, 46.6525230, -71.2678833, 71.9025269

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0753422, upper bound: 47.0541664
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0804335, upper bound: 47.0602312
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0804335, upper bound: 47.0605102
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -12.5650520, 43.4993477, -10.9450626, 37.6084023, -50.1734505, 54.2430267
1: -14.8664980, 50.5900497, -12.8934517, 43.6690178, -58.5025864, 63.1955566
2: -15.4648190, 49.4217529, -13.4669304, 42.6933708, -58.1269798, 62.6311951
3: -22.7446213, 53.2128296, -19.7413235, 45.9990692, -68.6559677, 72.6535263
4: -24.8334503, 47.7021942, -21.6055470, 41.2416763, -66.0175018, 69.1529007

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0753422, upper bound: 47.0541664
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0806644, upper bound: 47.0596186
time: 0.50 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0491549, upper bound: 47.0451427
time: 0.47 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -12.6311121, 43.6649780, -12.3291254, 42.5519447, -54.9989281, 55.8063965
1: -14.9605846, 50.7695885, -14.5488901, 49.4961662, -64.1635666, 65.0228195
2: -15.5325613, 49.6204185, -15.1856499, 48.3492012, -63.6156158, 64.5358963
3: -22.8752594, 53.4159927, -22.3086872, 52.1212234, -74.6669540, 75.3869171
4: -24.9373646, 47.8990097, -24.4192924, 46.6525230, -71.3708801, 72.0998688

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0753422, upper bound: 47.0599607
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0902454, upper bound: 47.0657722
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0902454, upper bound: 47.0660411
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -12.6311121, 43.6649780, -10.9450626, 37.6084023, -50.2395134, 54.4098129
1: -14.9605846, 50.7695885, -12.8934517, 43.6690178, -58.6017342, 63.3748283
2: -15.5325613, 49.6204185, -13.4669304, 42.6933708, -58.1903648, 62.8297882
3: -22.8752594, 53.4159927, -19.7413235, 45.9990692, -68.7855301, 72.8543930
4: -24.9373646, 47.8990097, -21.6055470, 41.2416763, -66.1204910, 69.3502502

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0487844, upper bound: 47.0518790
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0904231, upper bound: 47.0652454
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0489149, upper bound: 47.0452380
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -10.7382069, 37.0713997, -12.3291254, 42.5519447, -53.0927811, 49.4005203
1: -12.6549892, 43.0521164, -14.5488901, 49.4961662, -61.8628426, 57.5618172
2: -13.2225256, 42.0720711, -15.1856499, 48.3492012, -61.3186111, 57.2140617
3: -19.3875694, 45.3159103, -22.3086872, 52.1212234, -71.2133255, 67.5236130
4: -21.2321262, 40.6393242, -24.4192924, 46.6525230, -67.7280426, 64.9982071

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0390412
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0393201
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -10.7382069, 37.0713997, -10.9450626, 37.6084023, -48.3466110, 48.0164642
1: -12.6549892, 43.0521164, -12.8934517, 43.6690178, -56.3010292, 55.9138298
2: -13.2225256, 42.0720711, -13.4669304, 42.6933708, -55.8933601, 55.5079727
3: -19.3875694, 45.3159103, -19.7413235, 45.9990692, -65.3319016, 64.9910812
4: -21.2321262, 40.6393242, -21.6055470, 41.2416763, -62.4738007, 62.2448730

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0390412
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0393201
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -10.9528446, 37.6870155, -12.3291254, 42.5519447, -53.3107796, 50.0161362
1: -12.9132843, 43.7553520, -14.5488901, 49.4961662, -62.1257629, 58.2688866
2: -13.4681273, 42.7748718, -15.1856499, 48.3492012, -61.5634689, 57.9159737
3: -19.7512875, 46.0691261, -22.3086872, 52.1212234, -71.5770340, 68.2782593
4: -21.6068287, 41.3166008, -24.4192924, 46.6525230, -68.1021957, 65.6747589

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0401747
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0404536
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -10.9528446, 37.6870155, -10.9450626, 37.6084023, -48.5612450, 48.6320801
1: -12.9132843, 43.7553520, -12.8934517, 43.6690178, -56.5639534, 56.6208992
2: -13.4681273, 42.7748718, -13.4669304, 42.6933708, -56.1382179, 56.2098885
3: -19.7512875, 46.0691261, -19.7413235, 45.9990692, -65.6956100, 65.7457199
4: -21.6068287, 41.3166008, -21.6055470, 41.2416763, -62.8485031, 62.9221497

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0401747
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0404536
time: 0.86 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.85 seconds
NS_A1_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1656192, upper bound: 47.1628866
NS_A1_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1714059, upper bound: 47.1630303
NS_A1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1656192, upper bound: 47.1715351
NS_A1_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1714059, upper bound: 47.1711616
NS_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1719834, upper bound: 47.1599844
NS_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1713122, upper bound: 47.1629633
NS_A1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1422733, upper bound: 47.1634553
NS_A1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1722675, upper bound: 47.1706185
NS_A1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1712657, upper bound: 47.1705583
NS_A1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1725106, upper bound: 47.1698398
NS_A1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1733865, upper bound: 47.1690427
NS_A1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1733629, upper bound: 47.1688688
NS_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1695583, upper bound: 47.1655877
NS_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1689582, upper bound: 47.1689582
NS_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1678896, upper bound: 47.1672706
NS_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1678676, upper bound: 47.1678676
NS_A1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1118286, upper bound: 47.0672008
NS_A1_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1118286, upper bound: 47.0674529
NS_A1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1118286, upper bound: 47.0672009
NS_A1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1118286, upper bound: 47.0674529
NS_A1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1118286, upper bound: 47.0676431
NS_A1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1118286, upper bound: 47.0677933
NS_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1118286, upper bound: 47.0676431
NS_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.1118286, upper bound: 47.0677933
NS_A1_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0963289, upper bound: 47.0651248
NS_A1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0963289, upper bound: 47.0651248
NS_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0963289, upper bound: 47.0653220
NS_A1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0963289, upper bound: 47.0653220
NS_A1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0964260, upper bound: 47.0653976
NS_A1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0964260, upper bound: 47.0653976
NS_A1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0964260, upper bound: 47.0655185
NS_A1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0964260, upper bound: 47.0655185
NS_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0672009, upper bound: 47.1118286
NS_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0672009, upper bound: 47.1127253
NS_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0672009, upper bound: 47.1118286
NS_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0672009, upper bound: 47.1127253
NS_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0672009, upper bound: 47.1118286
NS_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0672009, upper bound: 47.1127253
NS_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0672009, upper bound: 47.1118286
NS_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0672009, upper bound: 47.1127253
NS_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0963289
NS_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0963289
NS_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0969853
NS_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0969853
NS_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0964260
NS_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0964260
NS_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0970870
NS_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0651248, upper bound: 47.0963289
NS_A2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0804335, upper bound: 47.0602312
NS_A2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0804335, upper bound: 47.0605102
NS_A2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0806644, upper bound: 47.0596186
NS_A2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0491549, upper bound: 47.0451427
NS_A2_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0902454, upper bound: 47.0657722
NS_A2_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0902454, upper bound: 47.0660411
NS_A2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0904231, upper bound: 47.0652454
NS_A2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0489149, upper bound: 47.0452380
NS_A2_B2_A2_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0390412
NS_A2_B2_A2_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0393201
NS_A2_B2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0390412
NS_A2_B2_A2_A1_B2_B2, status: Status.VERIFIED, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0390412, upper bound: 47.0393201
NS_A2_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0401747
NS_A2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0404536
NS_A2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0401747
NS_A2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.85
Output dim: 4, lower bound: -47.0393201, upper bound: 47.0404536

## BFS NS instance: NS_A1_B1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.6360275, 6.8702812, -3.4587708, 12.8446541, -14.4806805, 10.3290510
1: -1.8298423, 7.9820704, -3.9751954, 14.8369598, -16.6668015, 11.9572649
2: -2.1300311, 7.6280408, -4.3432884, 14.4518232, -16.5818539, 11.9713287
3: -2.9902825, 8.4492741, -6.2488904, 15.5703440, -18.5606251, 14.6981630
4: -3.9808242, 6.9692512, -7.2941327, 13.7042961, -17.6851196, 14.2633839

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1618329, upper bound: 47.1587063
time: 0.46 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1656119, upper bound: 47.1627768
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1656192, upper bound: 47.1599506
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1656192, upper bound: 47.1628866
time: 0.46 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.7392552, 7.2622151, -3.4587708, 12.8446541, -14.5839090, 10.7209835
1: -1.9530585, 8.4320698, -3.9751954, 14.8369598, -16.7900181, 12.4072647
2: -2.2601962, 8.0808363, -4.3432884, 14.4518232, -16.7120171, 12.4241247
3: -3.1773281, 8.9169788, -6.2488904, 15.5703440, -18.7476673, 15.1658688
4: -4.1860075, 7.4027514, -7.2941327, 13.7042961, -17.8903027, 14.6968842

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1670817, upper bound: 47.1586790
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1656119, upper bound: 47.1628891
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1714060, upper bound: 47.1600664
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1714060, upper bound: 47.1630303
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.7554306, 7.2911935, -4.1416893, 14.9934130, -16.7488403, 11.4328823
1: -1.9625205, 8.4706192, -4.7075543, 17.3641205, -19.3266411, 13.1781731
2: -2.2833903, 8.1021166, -5.2026291, 16.8603134, -19.1437035, 13.3047457
3: -3.1967328, 8.9663429, -7.3769655, 18.2618446, -21.4585762, 16.3433075
4: -4.2272167, 7.4234290, -8.6624832, 16.0341702, -20.2613869, 16.0859108

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1618329, upper bound: 47.1652913
time: 0.52 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1656119, upper bound: 47.1710492
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1698229, upper bound: 47.1541549
time: 0.44 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1627129, upper bound: 47.1530306
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.8622816, 7.7005429, -4.1416893, 14.9934130, -16.8556938, 11.8422318
1: -2.0885460, 8.9416027, -4.7075543, 17.3641205, -19.4526672, 13.6491566
2: -2.4184344, 8.5747175, -5.2026291, 16.8603134, -19.2787476, 13.7773466
3: -3.3904381, 9.4542942, -7.3769655, 18.2618446, -21.6522827, 16.8312607
4: -4.4408617, 7.8761816, -8.6624832, 16.0341702, -20.4750309, 16.5386600

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1618329, upper bound: 47.1649004
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1713987, upper bound: 47.1711341
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1757303, upper bound: 47.1711616
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -2.8822157, 11.0232143, -2.9533756, 11.2134895, -14.0957050, 13.9765892
1: -3.2299409, 12.7959824, -3.4005156, 12.9355078, -16.1654491, 16.1964970
2: -3.6806459, 12.3376141, -3.7272532, 12.5957603, -16.2764053, 16.0648670
3: -5.1305866, 13.4789286, -5.3433857, 13.5772228, -18.7078094, 18.8223095
4: -6.3740869, 11.5627041, -6.3452916, 11.8728409, -18.2469254, 17.9079933

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1677624, upper bound: 47.1563470
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1641792, upper bound: 47.1560469
time: 0.47 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -2.8822157, 11.0232143, -3.3701146, 12.6267567, -15.5089712, 14.3933287
1: -3.2299409, 12.7959824, -3.8883009, 14.5844812, -17.8144169, 16.6842804
2: -3.6806459, 12.3376141, -4.2331519, 14.2138824, -17.8945274, 16.5707664
3: -5.1305866, 13.4789286, -6.1186604, 15.3123894, -20.4429703, 19.5975895
4: -6.3740869, 11.5627041, -7.1457701, 13.4530869, -19.8271732, 18.7084732

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.8743403, 11.0149927, -4.1940603, 15.1722717, -18.0466118, 15.2090530
1: -3.2230463, 12.7823753, -4.7692404, 17.5731640, -20.7962112, 17.5516167
2: -3.6752691, 12.3271027, -5.2681537, 17.0648670, -20.7401352, 17.5952568
3: -5.1195831, 13.4734640, -7.4714885, 18.4803181, -23.5999012, 20.9449520
4: -6.3762746, 11.5466337, -8.7648716, 16.2336521, -22.6099262, 20.3115044

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1422733, upper bound: 47.1634553
time: 0.50 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1422543, upper bound: 47.1634553
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1422733, upper bound: 47.1634553
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1422733, upper bound: 47.1634553
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.9984341, 11.4355268, -4.1940603, 15.1722717, -18.1707058, 15.6295872
1: -3.3765814, 13.2645197, -4.7692404, 17.5731640, -20.9497452, 18.0337582
2: -3.8208203, 12.8097439, -5.2681537, 17.0648670, -20.8856869, 18.0778980
3: -5.3483610, 13.9750071, -7.4714885, 18.4803181, -23.8286781, 21.4464951
4: -6.5992360, 12.0168743, -8.7648716, 16.2336521, -22.8328876, 20.7817459

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1713048, upper bound: 47.1705013
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1722469, upper bound: 47.1706185
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1722469, upper bound: 47.1706185
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.7060931, 13.6615725, -4.0885248, 14.8552752, -18.5613670, 17.7500973
1: -4.1842017, 15.8413296, -4.6401472, 17.2174129, -21.4016151, 20.4814758
2: -4.6780243, 15.3407583, -5.1397719, 16.6970844, -21.3751087, 20.4805260
3: -6.5952477, 16.6680794, -7.2818685, 18.1143341, -24.7095814, 23.9499474
4: -7.9048872, 14.5074120, -8.5971518, 15.8510361, -23.7559242, 23.1045647

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1728410, upper bound: 47.1713689
time: 0.47 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1431557, upper bound: 47.1708823
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.8213358, 14.0880947, -4.0885248, 14.8552752, -18.6766109, 18.1766148
1: -4.3352304, 16.3396053, -4.6401472, 17.2174129, -21.5526409, 20.9797516
2: -4.8144999, 15.8319511, -5.1397719, 16.6970844, -21.5115852, 20.9717216
3: -6.8308606, 17.1838608, -7.2818685, 18.1143341, -24.9451942, 24.4657288
4: -8.1277628, 14.9757900, -8.5971518, 15.8510361, -23.9787979, 23.5729408

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1724187, upper bound: 47.1707872
time: 0.45 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1704507, upper bound: 47.1704507
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.2613163, 8.9881048, -6.1567283, 21.2260284, -23.4873428, 15.1448326
1: -2.5182602, 10.4307280, -7.0164323, 24.6163025, -27.1345596, 17.4471607
2: -2.9104450, 10.0213757, -7.6080451, 23.9474525, -26.8578968, 17.6294212
3: -4.0565658, 11.0031033, -10.8448687, 25.8670139, -29.9235783, 21.8479710
4: -5.1942396, 9.2732372, -12.3262997, 22.9762783, -28.1705170, 21.5995312

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1731392, upper bound: 47.1653697
time: 0.45 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1729537, upper bound: 47.1657614
time: 0.51 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.8189449, 13.9697170, -6.2036738, 21.3862705, -25.2052116, 20.1733913
1: -4.3157291, 16.1961479, -7.0725803, 24.8031158, -29.1188412, 23.2687263
2: -4.8130074, 15.6852398, -7.6664505, 24.1312027, -28.9442101, 23.3516903
3: -6.7913394, 17.0517178, -10.9312239, 26.0610123, -32.8523521, 27.9829407
4: -8.1063042, 14.8605747, -12.4161844, 23.1557178, -31.2620220, 27.2767601

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1731176, upper bound: 47.1652099
time: 0.49 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1730203, upper bound: 47.1655762
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -6.0737481, 20.9647923, -3.8243711, 14.0340357, -20.1077805, 24.7891636
1: -6.9062757, 24.3164692, -4.3346252, 16.2694664, -23.1757431, 28.6510944
2: -7.5086946, 23.6424351, -4.8191795, 15.7711840, -23.2798786, 28.4616146
3: -10.6806793, 25.5491333, -6.8182573, 17.1192474, -27.7999249, 32.3673859
4: -12.1793528, 22.6682663, -8.1150179, 14.9357138, -27.1150665, 30.7832832

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1669456, upper bound: 47.1702450
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1675100, upper bound: 47.1702989
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -6.0737481, 20.9647923, -3.9502518, 14.4872408, -20.5609894, 24.9150448
1: -6.9062757, 24.3164692, -4.4985547, 16.7998276, -23.7061024, 28.8150196
2: -7.5086946, 23.6424351, -4.9678092, 16.2937374, -23.8024330, 28.6102448
3: -10.6806793, 25.5491333, -7.0720854, 17.6708908, -28.3515682, 32.6212120
4: -12.1793528, 22.6682663, -8.3564253, 15.4357233, -27.6150761, 31.0246925

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1661495, upper bound: 47.1719326
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1667341, upper bound: 47.1719644
time: 0.49 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -5.7786617, 20.0360603, -5.1154051, 17.9620304, -23.7406921, 25.1514664
1: -6.5654030, 23.2357273, -5.8550353, 20.7860451, -27.3514462, 29.0907574
2: -7.1518602, 22.5830002, -6.3262277, 20.2429676, -27.3948250, 28.9092274
3: -10.1706581, 24.4070473, -9.0907803, 21.7931328, -31.9637833, 33.4978256
4: -11.6227131, 21.6338787, -10.2859144, 19.3818359, -31.0045471, 31.9197884

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673618, upper bound: 47.1672706
time: 0.51 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673618, upper bound: 47.1672706
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -6.0737481, 20.9647923, -6.1029205, 21.0067959, -27.0805435, 27.0677128
1: -6.9062757, 24.3164692, -6.9584203, 24.3565216, -31.2627964, 31.2748890
2: -7.5086946, 23.6424351, -7.5369096, 23.6974564, -31.2061501, 31.1793442
3: -10.6806793, 25.5491333, -10.7420139, 25.5904827, -36.2711639, 36.2911453
4: -12.1793528, 22.6682663, -12.1946831, 22.7528343, -34.9321861, 34.8629417

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673618, upper bound: 47.1678676
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1673618, upper bound: 47.1678676
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -3.0475309, 11.4952679, -12.0366526, 41.6751823, -44.5047569, 23.5319176
1: -3.4280396, 13.3304625, -14.1835155, 48.4934273, -51.6585846, 27.5139771
2: -3.8826449, 12.8824883, -14.8519659, 47.3338814, -50.9882774, 27.7344513
3: -5.4488935, 14.0269852, -21.7753677, 51.0566101, -56.2675018, 35.8023491
4: -6.6468649, 12.1436949, -23.9091930, 45.6596222, -52.2225418, 36.0528755

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -3.0475309, 11.4952679, -12.1057711, 41.8302841, -44.6612663, 23.6010399
1: -3.4280396, 13.3304625, -14.2765646, 48.6610146, -51.8251457, 27.6070271
2: -3.8826449, 12.8824883, -14.9221649, 47.5215149, -51.1751060, 27.8046513
3: -5.4488935, 14.0269852, -21.9043694, 51.2496605, -56.4582291, 35.9313507
4: -6.6468649, 12.1436949, -24.0188999, 45.8466415, -52.4097786, 36.1625862

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -3.0475309, 11.4952679, -10.5061388, 36.2615242, -39.3078842, 22.0014076
1: -3.4280396, 13.3304625, -12.3566360, 42.1188049, -45.5440598, 25.6870995
2: -3.8826449, 12.8824883, -12.9574594, 41.1499481, -45.0325928, 25.8399467
3: -5.4488935, 14.0269852, -18.9606285, 44.3618164, -49.8095703, 32.9876137
4: -6.6468649, 12.1436949, -20.8277321, 39.7396584, -46.3865242, 32.9714165

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -3.0475309, 11.4952679, -10.7305155, 36.9105186, -39.9526443, 22.2257843
1: -3.4280396, 13.3304625, -12.6277876, 42.8617096, -46.2897453, 25.9582500
2: -3.8826449, 12.8824883, -13.2123547, 41.8890800, -45.7717247, 26.0948410
3: -5.4488935, 14.0269852, -19.3433857, 45.1549759, -50.6033096, 33.3703613
4: -6.6468649, 12.1436949, -21.2191257, 40.4535484, -47.1004066, 33.3628197

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -3.6656094, 13.4249163, -12.0366526, 41.6751823, -45.1324425, 25.4615669
1: -4.1421041, 15.5496635, -14.1835155, 48.4934273, -52.3779526, 29.7331791
2: -4.6192713, 15.0691986, -14.8519659, 47.3338814, -51.7271957, 29.9211636
3: -6.5204768, 16.3473377, -21.7753677, 51.0566101, -57.3371277, 38.1227036
4: -7.7538829, 14.2926893, -23.9091930, 45.6596222, -53.3301392, 38.2018776

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -3.6656094, 13.4249163, -12.1057711, 41.8302841, -45.2889557, 25.5306873
1: -4.1421041, 15.5496635, -14.2765646, 48.6610146, -52.5445175, 29.8262272
2: -4.6192713, 15.0691986, -14.9221649, 47.5215149, -51.9140244, 29.9913616
3: -6.5204768, 16.3473377, -21.9043694, 51.2496605, -57.5278549, 38.2517014
4: -7.7538829, 14.2926893, -24.0188999, 45.8466415, -53.5173798, 38.3115845

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -3.6656094, 13.4249163, -10.5061388, 36.2615242, -39.9271317, 23.9310551
1: -4.1421041, 15.5496635, -12.3566360, 42.1188049, -46.2609100, 27.9062996
2: -4.6192713, 15.0691986, -12.9574594, 41.1499481, -45.7692184, 28.0266571
3: -6.5204768, 16.3473377, -18.9606285, 44.3618164, -50.8791962, 35.3079681
4: -7.7538829, 14.2926893, -20.8277321, 39.7396584, -47.4935417, 35.1204185

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0914040, upper bound: 47.0621750
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3.6656094, 13.4249163, -10.7305155, 36.9105186, -40.5761261, 24.1554317
1: -4.1421041, 15.5496635, -12.6277876, 42.8617096, -47.0038147, 28.1774521
2: -4.6192713, 15.0691986, -13.2123547, 41.8890800, -46.5083504, 28.2815533
3: -6.5204768, 16.3473377, -19.3433857, 45.1549759, -51.6729355, 35.6907120
4: -7.7538829, 14.2926893, -21.2191257, 40.4535484, -48.2074280, 35.5118141

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0914040, upper bound: 47.0621750
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -2.4840326, 9.2791185, -12.0366526, 41.6751823, -43.9403648, 21.3157692
1: -2.8140178, 10.6496401, -14.1835155, 48.4934273, -51.0388718, 24.8331566
2: -3.1398907, 10.3862915, -14.8519659, 47.3338814, -50.2652054, 25.2382584
3: -4.4031024, 11.2005587, -21.7753677, 51.0566101, -55.2462845, 32.9759254
4: -5.3245440, 9.8329000, -23.9091930, 45.6596222, -50.9723549, 33.7420921

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -2.4840326, 9.2791185, -10.5061388, 36.2615242, -38.7434883, 19.7852573
1: -2.8140178, 10.6496401, -12.3566360, 42.1188049, -44.9243469, 23.0062752
2: -3.1398907, 10.3862915, -12.9574594, 41.1499481, -44.2898369, 23.3437500
3: -4.4031024, 11.2005587, -18.9606285, 44.3618164, -48.7649117, 30.1611862
4: -5.3245440, 9.8329000, -20.8277321, 39.7396584, -45.0642014, 30.6606331

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -2.4840326, 9.2791185, -12.1057711, 41.8302841, -44.0968742, 21.3848896
1: -2.8140178, 10.6496401, -14.2765646, 48.6610146, -51.2054367, 24.9262028
2: -3.1398907, 10.3862915, -14.9221649, 47.5215149, -50.4520340, 25.3084564
3: -4.4031024, 11.2005587, -21.9043694, 51.2496605, -55.4370117, 33.1049271
4: -5.3245440, 9.8329000, -24.0188999, 45.8466415, -51.1595917, 33.8517990

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -2.4840326, 9.2791185, -10.7305155, 36.9105186, -39.3882523, 20.0096340
1: -2.8140178, 10.6496401, -12.6277876, 42.8617096, -45.6700325, 23.2774277
2: -3.1398907, 10.3862915, -13.2123547, 41.8890800, -45.0289688, 23.5986462
3: -4.4031024, 11.2005587, -19.3433857, 45.1549759, -49.5580711, 30.5439453
4: -5.3245440, 9.8329000, -21.2191257, 40.4535484, -45.7780914, 31.0520248

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -3.0158601, 11.0064144, -12.0366526, 41.6751823, -44.4784584, 23.0430660
1: -3.4244437, 12.6325045, -14.1835155, 48.4934273, -51.6520615, 26.8160210
2: -3.7862558, 12.3272820, -14.8519659, 47.3338814, -50.9127655, 27.1792431
3: -5.3343902, 13.2565231, -21.7753677, 51.0566101, -56.1751595, 35.0318909
4: -6.2886052, 11.7358017, -23.9091930, 45.6596222, -51.9383888, 35.6449890

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -3.0158601, 11.0064144, -10.5061388, 36.2615242, -39.2773857, 21.5125542
1: -3.4244437, 12.6325045, -12.3566360, 42.1188049, -45.5375366, 24.9891396
2: -3.7862558, 12.3272820, -12.9574594, 41.1499481, -44.9362030, 25.2847385
3: -5.3343902, 13.2565231, -18.9606285, 44.3618164, -49.6962013, 32.2171516
4: -6.2886052, 11.7358017, -20.8277321, 39.7396584, -46.0282631, 32.5635300

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -3.0158601, 11.0064144, -12.1057711, 41.8302841, -44.6349678, 23.1121864
1: -3.4244437, 12.6325045, -14.2765646, 48.6610146, -51.8186264, 26.9090691
2: -3.7862558, 12.3272820, -14.9221649, 47.5215149, -51.0995941, 27.2494431
3: -5.3343902, 13.2565231, -21.9043694, 51.2496605, -56.3658867, 35.1608925
4: -6.2886052, 11.7358017, -24.0188999, 45.8466415, -52.1256332, 35.7546997

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3.0158601, 11.0064144, -10.7305155, 36.9105186, -39.9263458, 21.7369308
1: -3.4244437, 12.6325045, -12.6277876, 42.8617096, -46.2832222, 25.2602921
2: -3.7862558, 12.3272820, -13.2123547, 41.8890800, -45.6753349, 25.5396328
3: -5.3343902, 13.2565231, -19.3433857, 45.1549759, -50.4893608, 32.5999069
4: -6.2886052, 11.7358017, -21.2191257, 40.4535484, -46.7421532, 32.9549255

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -12.5650520, 43.4993477, -3.0475309, 11.4952679, -24.0603199, 46.3301201
1: -14.8664980, 50.5900497, -3.4280396, 13.3304625, -28.1969604, 53.7560501
2: -15.4648190, 49.4217529, -3.8826449, 12.8824883, -28.3473053, 53.0762711
3: -22.7446213, 53.2128296, -5.4488935, 14.0269852, -36.7716064, 58.4223442
4: -24.8334503, 47.7021942, -6.6468649, 12.1436949, -36.9771385, 54.2639351

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -12.6311121, 43.6649780, -3.0475309, 11.4952679, -24.1263771, 46.4969063
1: -14.9605846, 50.7695885, -3.4280396, 13.3304625, -28.2910461, 53.9353180
2: -15.5325613, 49.6204185, -3.8826449, 12.8824883, -28.4150505, 53.2748642
3: -22.8752594, 53.4159927, -5.4488935, 14.0269852, -36.9022446, 58.6232300
4: -24.9373646, 47.8990097, -6.6468649, 12.1436949, -37.0810585, 54.4612541

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -10.7382069, 37.0713997, -3.0475309, 11.4952679, -22.2334728, 40.1152229
1: -12.6549892, 43.0521164, -3.4280396, 13.3304625, -25.9854507, 46.4743118
2: -13.2225256, 42.0720711, -3.8826449, 12.8824883, -26.1050129, 45.9530525
3: -19.3875694, 45.3159103, -5.4488935, 14.0269852, -33.4145546, 50.7599144
4: -21.2321262, 40.6393242, -6.6468649, 12.1436949, -33.3758202, 47.2861862

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -10.9528446, 37.6870155, -3.0475309, 11.4952679, -22.4481125, 40.7280197
1: -12.9132843, 43.7553520, -3.4280396, 13.3304625, -26.2437477, 47.1813889
2: -13.4681273, 42.7748718, -3.8826449, 12.8824883, -26.3506107, 46.6549644
3: -19.7512875, 46.0691261, -5.4488935, 14.0269852, -33.7782707, 51.5145683
4: -21.6068287, 41.3166008, -6.6468649, 12.1436949, -33.7505226, 47.9634666

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -12.5650520, 43.4993477, -3.6656094, 13.4249163, -25.9899673, 46.9578056
1: -14.8664980, 50.5900497, -4.1421041, 15.5496635, -30.4161606, 54.4754181
2: -15.4648190, 49.4217529, -4.6192713, 15.0691986, -30.5340176, 53.8151894
3: -22.7446213, 53.2128296, -6.5204768, 16.3473377, -39.0919571, 59.4919701
4: -24.8334503, 47.7021942, -7.7538829, 14.2926893, -39.1261406, 55.3715363

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -12.6311121, 43.6649780, -3.6656094, 13.4249163, -26.0560284, 47.1245995
1: -14.9605846, 50.7695885, -4.1421041, 15.5496635, -30.5102482, 54.6546898
2: -15.5325613, 49.6204185, -4.6192713, 15.0691986, -30.6017609, 54.0137825
3: -22.8752594, 53.4159927, -6.5204768, 16.3473377, -39.2225952, 59.6928558
4: -24.9373646, 47.8990097, -7.7538829, 14.2926893, -39.2300529, 55.5688515

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -10.7382069, 37.0713997, -3.6656094, 13.4249163, -24.1631241, 40.7370071
1: -12.6549892, 43.0521164, -4.1421041, 15.5496635, -28.2046490, 47.1936836
2: -13.2225256, 42.0720711, -4.6192713, 15.0691986, -28.2917233, 46.6913414
3: -19.3875694, 45.3159103, -6.5204768, 16.3473377, -35.7349091, 51.8295403
4: -21.2321262, 40.6393242, -7.7538829, 14.2926893, -35.5248146, 48.3932076

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -10.9528446, 37.6870155, -3.6656094, 13.4249163, -24.3777618, 41.3526192
1: -12.9132843, 43.7553520, -4.1421041, 15.5496635, -28.4629459, 47.8974571
2: -13.4681273, 42.7748718, -4.6192713, 15.0691986, -28.5373230, 47.3938828
3: -19.7512875, 46.0691261, -6.5204768, 16.3473377, -36.0986252, 52.5841942
4: -21.6068287, 41.3166008, -7.7538829, 14.2926893, -35.8995171, 49.0704842

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -12.5650520, 43.4993477, -2.4840326, 9.2791185, -21.8441696, 45.7657280
1: -14.8664980, 50.5900497, -2.8140178, 10.6496401, -25.5161381, 53.1363373
2: -15.4648190, 49.4217529, -3.1398907, 10.3862915, -25.8511105, 52.3531990
3: -22.7446213, 53.2128296, -4.4031024, 11.2005587, -33.9451790, 57.4011307
4: -24.8334503, 47.7021942, -5.3245440, 9.8329000, -34.6663513, 53.0137482

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -10.7382069, 37.0713997, -2.4840326, 9.2791185, -20.0173264, 39.5508308
1: -12.6549892, 43.0521164, -2.8140178, 10.6496401, -23.3046265, 45.8546028
2: -13.2225256, 42.0720711, -3.1398907, 10.3862915, -23.6088181, 45.2119598
3: -19.3875694, 45.3159103, -4.4031024, 11.2005587, -30.5881271, 49.7190056
4: -21.2321262, 40.6393242, -5.3245440, 9.8329000, -31.0650253, 45.9638672

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -12.6311121, 43.6649780, -2.4840326, 9.2791185, -21.9102306, 45.9325180
1: -14.9605846, 50.7695885, -2.8140178, 10.6496401, -25.6102238, 53.3156090
2: -15.5325613, 49.6204185, -3.1398907, 10.3862915, -25.9188538, 52.5517921
3: -22.8752594, 53.4159927, -4.4031024, 11.2005587, -34.0758171, 57.6020126
4: -24.9373646, 47.8990097, -5.3245440, 9.8329000, -34.7702637, 53.2110672

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -10.9528446, 37.6870155, -2.4840326, 9.2791185, -20.2319641, 40.1636276
1: -12.9132843, 43.7553520, -2.8140178, 10.6496401, -23.5629196, 46.5616760
2: -13.4681273, 42.7748718, -3.1398907, 10.3862915, -23.8544197, 45.9147606
3: -19.7512875, 46.0691261, -4.4031024, 11.2005587, -30.9518471, 50.4722214
4: -21.6068287, 41.3166008, -5.3245440, 9.8329000, -31.4397278, 46.6411438

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -12.5650520, 43.4993477, -3.0158601, 11.0064144, -23.5714664, 46.3038216
1: -14.8664980, 50.5900497, -3.4244437, 12.6325045, -27.4990025, 53.7495270
2: -15.4648190, 49.4217529, -3.7862558, 12.3272820, -27.7920971, 53.0007591
3: -22.7446213, 53.2128296, -5.3343902, 13.2565231, -36.0011444, 58.3300056
4: -24.8334503, 47.7021942, -6.2886052, 11.7358017, -36.5692520, 53.9797859

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -10.7382069, 37.0713997, -3.0158601, 11.0064144, -21.7446213, 40.0872574
1: -12.6549892, 43.0521164, -3.4244437, 12.6325045, -25.2874928, 46.4677925
2: -13.2225256, 42.0720711, -3.7862558, 12.3272820, -25.5498047, 45.8583260
3: -19.3875694, 45.3159103, -5.3343902, 13.2565231, -32.6440926, 50.6502953
4: -21.2321262, 40.6393242, -6.2886052, 11.7358017, -32.9679260, 46.9279289

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -12.6311121, 43.6649780, -3.0158601, 11.0064144, -23.6375256, 46.4706154
1: -14.9605846, 50.7695885, -3.4244437, 12.6325045, -27.5930901, 53.9287949
2: -15.5325613, 49.6204185, -3.7862558, 12.3272820, -27.8598423, 53.1993523
3: -22.8752594, 53.4159927, -5.3343902, 13.2565231, -36.1317825, 58.5308876
4: -24.9373646, 47.8990097, -6.2886052, 11.7358017, -36.6731644, 54.1771049

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -10.9528446, 37.6870155, -3.0158601, 11.0064144, -21.9592590, 40.7017212
1: -12.9132843, 43.7553520, -3.4244437, 12.6325045, -25.5457878, 47.1748657
2: -13.4681273, 42.7748718, -3.7862558, 12.3272820, -25.7954025, 46.5611267
3: -19.7512875, 46.0691261, -5.3343902, 13.2565231, -33.0078125, 51.4035110
4: -21.6068287, 41.3166008, -6.2886052, 11.7358017, -33.3426285, 47.6052055

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -12.5650520, 43.4993477, -12.0366526, 41.6751823, -54.0517540, 55.3504677
1: -14.8664980, 50.5900497, -14.1835155, 48.4934273, -63.0587311, 64.4743958
2: -15.4648190, 49.4217529, -14.8519659, 47.3338814, -62.5326500, 64.0071030
3: -22.7446213, 53.2128296, -21.7753677, 51.0566101, -73.4681396, 74.6523361
4: -24.8334503, 47.7021942, -23.9091930, 45.6596222, -70.2705383, 71.3852310

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_A1_B1_B1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0826855, upper bound: 47.0734406
time: 0.49 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_B1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0903079, upper bound: 47.0903079
time: 0.52 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -12.5650520, 43.4993477, -12.1057711, 41.8302841, -54.2082596, 55.4162407
1: -14.8664980, 50.5900497, -14.2765646, 48.6610146, -63.2252922, 64.5716019
2: -15.4648190, 49.4217529, -14.9221649, 47.5215149, -62.7194786, 64.0725861
3: -22.7446213, 53.2128296, -21.9043694, 51.2496605, -73.6588669, 74.7803955
4: -24.8334503, 47.7021942, -24.0188999, 45.8466415, -70.4577866, 71.4938507

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_A1_B1_B2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0826855, upper bound: 47.0735579
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_B2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0903079, upper bound: 47.1353222
time: 0.55 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -12.3930187, 42.9414673, -10.9450626, 37.6084023, -50.0014191, 53.6832466
1: -14.6621161, 49.9458618, -12.8934517, 43.6690178, -58.2967415, 62.5494499
2: -15.2581835, 48.7862930, -13.4669304, 42.6933708, -57.9187126, 61.9942360
3: -22.4425449, 52.5432625, -19.7413235, 45.9990692, -68.3537598, 71.9812012
4: -24.5252361, 47.0810127, -21.6055470, 41.2416763, -65.7090530, 68.5308380

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0774810, upper bound: 47.0583760
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0796095, upper bound: 47.0594674
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0796095, upper bound: 47.0596186
time: 0.53 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -12.7934980, 44.4138031, -10.9211092, 37.5276985, -50.3211861, 55.0561104
1: -15.1335058, 51.6793289, -12.8648634, 43.5754929, -58.6749153, 64.1612396
2: -15.7342596, 50.4522972, -13.4376774, 42.6008682, -58.3077736, 63.5527878
3: -23.1641121, 54.3491516, -19.6980228, 45.9023705, -68.9706879, 73.6606674
4: -25.2996006, 48.6298294, -21.5605259, 41.1518936, -66.3931198, 69.9842987

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0481000, upper bound: 47.0449915
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0481000, upper bound: 47.0451427
time: 0.49 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -12.6311121, 43.6649780, -12.0366526, 41.6751823, -54.1147690, 55.5172539
1: -14.9605846, 50.7695885, -14.1835155, 48.4934273, -63.1578751, 64.6536636
2: -15.5325613, 49.6204185, -14.8519659, 47.3338814, -62.5960350, 64.2057114
3: -22.8752594, 53.4159927, -21.7753677, 51.0566101, -73.5976944, 74.8532028
4: -24.9373646, 47.8990097, -23.9091930, 45.6596222, -70.3735352, 71.5825729

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_A2_B1_B1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1384200, upper bound: 47.1063424
time: 0.51 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_A2_B1_B1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1183163, upper bound: 47.0841290
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_B1_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1353223, upper bound: 47.1023891
time: 0.56 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -12.6311121, 43.6649780, -12.1057711, 41.8302841, -54.2712784, 55.5830345
1: -14.9605846, 50.7695885, -14.2765646, 48.6610146, -63.3244438, 64.7508698
2: -15.5325613, 49.6204185, -14.9221649, 47.5215149, -62.7828598, 64.2711868
3: -22.8752594, 53.4159927, -21.9043694, 51.2496605, -73.7884293, 74.9812622
4: -24.9373646, 47.8990097, -24.0188999, 45.8466415, -70.5607834, 71.6911926

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_A2_B1_B2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1384200, upper bound: 47.1210550
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_A2_B1_B2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1183163, upper bound: 47.0842067
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A1_A2_B1_B2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.1353223, upper bound: 47.1607664
time: 0.57 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -12.4578161, 43.1027145, -10.9450626, 37.6084023, -50.0662117, 53.8457260
1: -14.7548237, 50.1192398, -12.8934517, 43.6690178, -58.3945656, 62.7225075
2: -15.3249598, 48.9799385, -13.4669304, 42.6933708, -57.9809532, 62.1873627
3: -22.5707359, 52.7414322, -19.7413235, 45.9990692, -68.4809647, 72.1768188
4: -24.6275597, 47.2742424, -21.6055470, 41.2416763, -65.8104095, 68.7237625

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0844353, upper bound: 47.0590693
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0894354, upper bound: 47.0651038
time: 1.94 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0894354, upper bound: 47.0652454
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -12.8619518, 44.5868683, -10.9211092, 37.5276985, -50.3896446, 55.2302399
1: -15.2312450, 51.8682976, -12.8648634, 43.5754929, -58.7783012, 64.3502884
2: -15.8070869, 50.6671143, -13.4376774, 42.6008682, -58.3765984, 63.7655220
3: -23.2975712, 54.5610580, -19.6980228, 45.9023705, -69.1038513, 73.8719940
4: -25.4091778, 48.8405571, -21.5605259, 41.1518936, -66.5022125, 70.1937256

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0425310, upper bound: 47.0380404
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0478600, upper bound: 47.0450868
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -47.0478600, upper bound: 47.0451564
time: 0.48 seconds

## BFS NS instance: NS_A2_B2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -10.9528446, 37.6870155, -12.0366526, 41.6751823, -52.4266205, 49.7236595
1: -12.9132843, 43.7553520, -14.1835155, 48.4934273, -61.1200905, 57.8997345
2: -13.4681273, 42.7748718, -14.8519659, 47.3338814, -60.5438881, 57.5858040
3: -19.7512875, 46.0691261, -21.7753677, 51.0566101, -70.5077744, 67.7445297
4: -21.6068287, 41.3166008, -23.9091930, 45.6596222, -67.1048584, 65.1574631

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -10.9528446, 37.6870155, -12.1057711, 41.8302841, -52.5831299, 49.7927742
1: -12.9132843, 43.7553520, -14.2765646, 48.6610146, -61.2866516, 57.9969368
2: -13.4681273, 42.7748718, -14.9221649, 47.5215149, -60.7307167, 57.6512871
3: -19.7512875, 46.0691261, -21.9043694, 51.2496605, -70.6985016, 67.8725891
4: -21.6068287, 41.3166008, -24.0188999, 45.8466415, -67.2921066, 65.2660828

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -10.9528446, 37.6870155, -10.5061388, 36.2615242, -47.2143707, 48.1931534
1: -12.9132843, 43.7553520, -12.3566360, 42.1188049, -55.0055656, 56.0800819
2: -13.4681273, 42.7748718, -12.9574594, 41.1499481, -54.5890694, 55.6997070
3: -19.7512875, 46.0691261, -18.9606285, 44.3618164, -64.0498276, 64.9640274
4: -21.6068287, 41.3166008, -20.8277321, 39.7396584, -61.3452644, 62.1399727

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_A2_B2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -10.9528446, 37.6870155, -10.7305155, 36.9105186, -47.8633652, 48.4175301
1: -12.9132843, 43.7553520, -12.6277876, 42.8617096, -55.7512550, 56.3560143
2: -13.4681273, 42.7748718, -13.2123547, 41.8890800, -55.3274422, 55.9538765
3: -19.7512875, 46.0691261, -19.3433857, 45.1549759, -64.8435669, 65.3467026
4: -21.6068287, 41.3166008, -21.2191257, 40.4535484, -62.0586433, 62.5307159

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.03 + 413.87 = 416.90 seconds
