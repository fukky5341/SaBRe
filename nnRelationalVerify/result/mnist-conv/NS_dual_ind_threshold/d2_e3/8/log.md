## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.2940516


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.0033817, -4.6783876, -6.0033817, -4.6783876, -0.5389104, 0.5389103)
1: (-11.1173611, -9.8019371, -11.1173611, -9.8019371, -0.4993663, 0.4993664)
2: (6.1117640, 7.2846823, 6.1117640, 7.2846823, -0.4197077, 0.4197077)
3: (-4.7743611, -3.9336448, -4.7743611, -3.9336448, -0.3537687, 0.3537687)
4: (-12.3435221, -11.2221107, -12.3435221, -11.2221107, -0.3765505, 0.3765505)
5: (-13.7827396, -12.7530775, -13.7827396, -12.7530775, -0.3536179, 0.3536178)
6: (-10.9417925, -9.7292747, -10.9417925, -9.7292747, -0.5343599, 0.5343599)
7: (-1.7105432, -0.7294660, -1.7105432, -0.7294660, -0.3434693, 0.3434693)
8: (-0.6338139, 0.2915998, -0.6338139, 0.2915998, -0.3664252, 0.3664252)
9: (-10.0903921, -8.8831673, -10.0903921, -8.8831673, -0.5006785, 0.5006785)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.05 + 34.32 = 57.37 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3267234, upper bound: 0.3267232

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 675
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.41 seconds

### Candidate
type: A, layer: 3, pos: 675

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3157591, upper bound: 0.3215697
time: 4.66 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3215697, upper bound: 0.3215702
time: 3.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.38 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.38
Output dim: 2, lower bound: -0.3157591, upper bound: 0.3215697
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.38
Output dim: 2, lower bound: -0.3215697, upper bound: 0.3215702

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -5.9843969, -4.6783876, -5.9976172, -4.6783876, -0.5188112, 0.5323087
1: -11.1173601, -9.8143692, -11.1173611, -9.8056650, -0.4949231, 0.4852633
2: 6.1281986, 7.2846136, 6.1166925, 7.2846608, -0.4009869, 0.4132460
3: -4.7735462, -3.9379275, -4.7741184, -3.9350276, -0.3449365, 0.3318352
4: -12.3435221, -11.2320642, -12.3435230, -11.2250986, -0.3730469, 0.3654678
5: -13.7827053, -12.7533398, -13.7827301, -12.7531576, -0.3528756, 0.3515215
6: -10.9382477, -9.7311249, -10.9407291, -9.7298355, -0.5291035, 0.5288534
7: -1.7087009, -0.7294660, -1.7099912, -0.7294655, -0.3381132, 0.3417822
8: -0.6336932, 0.2913942, -0.6337786, 0.2915373, -0.3658352, 0.3657327
9: -10.0884876, -8.8977356, -10.0898247, -8.8875408, -0.4910133, 0.4830256

Time for backsubstitution: 8.92 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1376

Time for candidate selection: 0.42 seconds

### Candidate
type: B, layer: 3, pos: 172

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3074245, upper bound: 0.3166486
time: 3.44 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3074309, upper bound: 0.3132421
time: 3.58 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -5.9940472, -4.6570978, -5.9994783, -4.6783876, -0.5249091, 0.5715132
1: -11.1307716, -9.8080826, -11.1173592, -9.8044310, -0.5211368, 0.4898131
2: 6.1226683, 7.3025036, 6.1160021, 7.2846613, -0.4046220, 0.4539121
3: -4.7643957, -3.9593542, -4.7741356, -3.9437406, -0.3783069, 0.3251345
4: -12.3546038, -11.2261906, -12.3435211, -11.2238083, -0.3929626, 0.3695075
5: -13.7811842, -12.7561016, -13.7827311, -12.7542496, -0.3564539, 0.3504832
6: -10.9390812, -9.7272129, -10.9407167, -9.7297935, -0.5296214, 0.5331826
7: -1.7040806, -0.7313976, -1.7079186, -0.7294660, -0.3374673, 0.3500289
8: -0.6337004, 0.2905169, -0.6337795, 0.2911787, -0.3663979, 0.3655542
9: -10.1061983, -8.8909626, -10.0898628, -8.8863649, -0.5184667, 0.4878869

Time for backsubstitution: 8.41 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 172
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1376

Time for candidate selection: 0.41 seconds

### Candidate
type: B, layer: 3, pos: 172

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3132350, upper bound: 0.3166487
time: 6.32 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3132414, upper bound: 0.3132421
time: 3.44 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 18.59 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 18.59
Output dim: 2, lower bound: -0.3074245, upper bound: 0.3166486
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 18.59
Output dim: 2, lower bound: -0.3074309, upper bound: 0.3132421
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 18.59
Output dim: 2, lower bound: -0.3132350, upper bound: 0.3166487
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 18.59
Output dim: 2, lower bound: -0.3132414, upper bound: 0.3132421

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -5.9843969, -4.6783876, -5.9885421, -4.6783876, -0.5188112, 0.5203621
1: -11.1173601, -9.8143692, -11.1173592, -9.8331509, -0.4652040, 0.4852632
2: 6.1281986, 7.2846136, 6.1209068, 7.2841587, -0.4003996, 0.4001657
3: -4.7735462, -3.9379275, -4.7660427, -3.9350274, -0.3449364, 0.3175356
4: -12.3435221, -11.2320642, -12.3431702, -11.2262115, -0.3715030, 0.3651151
5: -13.7827053, -12.7533398, -13.7816887, -12.7539349, -0.3502647, 0.3501827
6: -10.9382477, -9.7311249, -10.9256191, -9.7302876, -0.5280259, 0.5100154
7: -1.7087009, -0.7294660, -1.7098670, -0.7377782, -0.3267936, 0.3414785
8: -0.6336932, 0.2913942, -0.6199317, 0.2915349, -0.3658350, 0.3430570
9: -10.0884876, -8.8977356, -10.0893555, -8.8898621, -0.4870204, 0.4820646

Time for backsubstitution: 8.38 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 172

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3074245, upper bound: 0.3132357
time: 3.92 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3074245, upper bound: 0.3132415
time: 3.57 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -5.9756870, -4.6783876, -5.9750681, -4.6505003, -0.5689588, 0.5231946
1: -11.1173592, -9.8272018, -11.2034912, -9.8388214, -0.4798996, 0.5798513
2: 6.1386156, 7.2844405, 6.1449943, 7.2967749, -0.4487019, 0.3957515
3: -4.7597613, -3.9379277, -4.7369084, -3.9170918, -0.3940425, 0.3126656
4: -12.3433113, -11.2325373, -12.3429394, -11.2256050, -0.3714457, 0.3643233
5: -13.7823725, -12.7566109, -13.7823744, -12.7620344, -0.3483757, 0.3570307
6: -10.9313745, -9.7317820, -10.9258556, -9.7027454, -0.5503953, 0.5163063
7: -1.7085238, -0.7368221, -1.7364714, -0.7448564, -0.3296264, 0.3717213
8: -0.6219349, 0.2913914, -0.6065993, 0.3261499, -0.4213545, 0.3469024
9: -10.0878048, -8.9009609, -10.0914783, -8.8962898, -0.4845721, 0.4869680

Time for backsubstitution: 8.35 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 900

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3107803
time: 4.09 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3052257, upper bound: 0.3110369
time: 3.41 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -5.9940472, -4.6570978, -5.9904089, -4.6783876, -0.5249091, 0.5595711
1: -11.1307716, -9.8080826, -11.1173592, -9.8319168, -0.4915719, 0.4898129
2: 6.1226683, 7.3025036, 6.1202173, 7.2841597, -0.4040350, 0.4409121
3: -4.7643957, -3.9593542, -4.7660618, -3.9437408, -0.3783067, 0.3108318
4: -12.3546038, -11.2261906, -12.3431702, -11.2249250, -0.3914878, 0.3691550
5: -13.7811842, -12.7561016, -13.7816887, -12.7550278, -0.3538463, 0.3491449
6: -10.9390812, -9.7272129, -10.9256763, -9.7302485, -0.5285444, 0.5143470
7: -1.7040806, -0.7313976, -1.7077980, -0.7377782, -0.3261477, 0.3497261
8: -0.6337004, 0.2905169, -0.6199336, 0.2911777, -0.3663977, 0.3428869
9: -10.1061983, -8.8909626, -10.0893936, -8.8886776, -0.5145561, 0.4869268

Time for backsubstitution: 9.03 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 172

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3132350, upper bound: 0.3132350
time: 4.85 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3132350, upper bound: 0.3132415
time: 5.97 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -5.9853306, -4.6570978, -5.9769330, -4.6505003, -0.5750501, 0.5623713
1: -11.1307716, -9.8209171, -11.2034922, -9.8375874, -0.5062790, 0.5843998
2: 6.1330881, 7.3023286, 6.1443052, 7.2967772, -0.4523346, 0.4365035
3: -4.7506061, -3.9593542, -4.7369270, -3.9258049, -0.4274505, 0.3059614
4: -12.3543949, -11.2266655, -12.3429394, -11.2243166, -0.3914518, 0.3683676
5: -13.7808485, -12.7593746, -13.7823734, -12.7631273, -0.3519690, 0.3559929
6: -10.9322300, -9.7278652, -10.9259138, -9.7027063, -0.5509076, 0.5206571
7: -1.7039058, -0.7387533, -1.7344060, -0.7448568, -0.3289802, 0.3800397
8: -0.6219416, 0.2905159, -0.6066012, 0.3257923, -0.4219358, 0.3467325
9: -10.1055212, -8.8941803, -10.0915155, -8.8950996, -0.5121680, 0.4918309

Time for backsubstitution: 9.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.43 seconds

### Candidate
type: A, layer: 3, pos: 900

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3088032, upper bound: 0.3107809
time: 3.59 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3110359, upper bound: 0.3110369
time: 3.49 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 16.60 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.60
Output dim: 2, lower bound: -0.3074245, upper bound: 0.3132357
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.60
Output dim: 2, lower bound: -0.3074245, upper bound: 0.3132415
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.60
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3107803
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.60
Output dim: 2, lower bound: -0.3052257, upper bound: 0.3110369
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 16.60
Output dim: 2, lower bound: -0.3132350, upper bound: 0.3132350
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 16.60
Output dim: 2, lower bound: -0.3132350, upper bound: 0.3132415
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 16.60
Output dim: 2, lower bound: -0.3088032, upper bound: 0.3107809
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 16.60
Output dim: 2, lower bound: -0.3110359, upper bound: 0.3110369

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.9753256, -4.6783876, -5.9885421, -4.6783876, -0.5068521, 0.5203621
1: -11.1173582, -9.8418522, -11.1173592, -9.8331509, -0.4652040, 0.4554037
2: 6.1324129, 7.2841110, 6.1209068, 7.2841587, -0.3873203, 0.3995806
3: -4.7654700, -3.9379277, -4.7660427, -3.9350274, -0.3306088, 0.3175356
4: -12.3431702, -11.2331810, -12.3431702, -11.2262115, -0.3711505, 0.3635248
5: -13.7816658, -12.7541170, -13.7816887, -12.7539349, -0.3489262, 0.3475563
6: -10.9231339, -9.7315836, -10.9256191, -9.7302876, -0.5091431, 0.5089391
7: -1.7085752, -0.7377777, -1.7098670, -0.7377782, -0.3264917, 0.3301589
8: -0.6198463, 0.2913914, -0.6199317, 0.2915349, -0.3431441, 0.3430569
9: -10.0880108, -8.9000616, -10.0893555, -8.8898621, -0.4860591, 0.4780526

Time for backsubstitution: 9.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1376

Time for candidate selection: 0.42 seconds

### Candidate
type: B, layer: 3, pos: 675

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3074245, upper bound: 0.3132313
time: 3.52 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3074245, upper bound: 0.3166493
time: 3.59 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.9618578, -4.6505003, -5.9885421, -4.6783876, -0.5162003, 0.5717106
1: -11.2034893, -9.8475361, -11.1173592, -9.8331509, -0.5671213, 0.4665196
2: 6.1564984, 7.2967281, 6.1209068, 7.2841587, -0.3994384, 0.4482635
3: -4.7363181, -3.9199934, -4.7660427, -3.9350274, -0.3356745, 0.3701607
4: -12.3429394, -11.2325630, -12.3431702, -11.2262115, -0.3711607, 0.3636399
5: -13.7823524, -12.7622166, -13.7816887, -12.7539349, -0.3563222, 0.3486736
6: -10.9233694, -9.7040339, -10.9256191, -9.7302876, -0.5207117, 0.5349310
7: -1.7351809, -0.7448583, -1.7098670, -0.7377782, -0.3596829, 0.3343359
8: -0.6065111, 0.3260055, -0.6199317, 0.2915349, -0.3625399, 0.3998999
9: -10.0901184, -8.9064827, -10.0893555, -8.8898621, -0.4913493, 0.4813172

Time for backsubstitution: 8.38 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1376

Time for candidate selection: 0.42 seconds

### Candidate
type: B, layer: 3, pos: 675

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3074245, upper bound: 0.3132305
time: 3.43 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3074245, upper bound: 0.3166492
time: 3.49 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.9526472, -4.6783876, -5.9640760, -4.6505003, -0.5546737, 0.5156684
1: -11.1017761, -9.8275452, -11.1976147, -9.8389158, -0.4741455, 0.5779973
2: 6.1481256, 7.2843719, 6.1485262, 7.2967567, -0.4422468, 0.3928236
3: -4.7566972, -3.9379983, -4.7356758, -3.9171185, -0.3889172, 0.3107283
4: -12.3069334, -11.2328310, -12.3280563, -11.2256985, -0.3322228, 0.3471280
5: -13.7816381, -12.7643986, -13.7821245, -12.7648878, -0.3396471, 0.3353364
6: -10.9289207, -9.7465906, -10.9250669, -9.7102222, -0.5412939, 0.4995270
7: -1.6897473, -0.7370784, -1.7292054, -0.7449310, -0.3187997, 0.3655775
8: -0.6213489, 0.2852368, -0.6064472, 0.3237333, -0.4178133, 0.3419231
9: -10.0713844, -8.9019098, -10.0851479, -8.8966284, -0.4677721, 0.4803818

Time for backsubstitution: 8.96 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1376

Time for candidate selection: 0.41 seconds

### Candidate
type: B, layer: 3, pos: 675

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3073636
time: 5.06 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3107803
time: 4.09 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.9722652, -4.6564512, -5.9723301, -4.6505003, -0.5741827, 0.5319760
1: -11.1140299, -9.8105927, -11.2022095, -9.8388720, -0.4793766, 0.5750376
2: 6.1423125, 7.2924485, 6.1479883, 7.2967687, -0.4518303, 0.3953921
3: -4.7563605, -3.9383416, -4.7356057, -3.9170990, -0.3887548, 0.3166853
4: -12.3396454, -11.1958008, -12.3415098, -11.2256460, -0.3483450, 0.4054809
5: -13.7831650, -12.7671728, -13.7823114, -12.7673073, -0.3749853, 0.3336818
6: -10.9648972, -9.7387333, -10.9256706, -9.7053032, -0.6007020, 0.5033130
7: -1.7023544, -0.7212458, -1.7342114, -0.7448730, -0.3251374, 0.3804896
8: -0.6337538, 0.2911882, -0.6065712, 0.3254251, -0.4383159, 0.3519128
9: -10.0835028, -8.8846493, -10.0897789, -8.8963757, -0.4724251, 0.5079224

Time for backsubstitution: 8.43 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1376

Time for candidate selection: 0.44 seconds

### Candidate
type: B, layer: 3, pos: 675

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3052257, upper bound: 0.3076180
time: 3.29 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3052257, upper bound: 0.3110369
time: 3.41 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.9849749, -4.6570978, -5.9904089, -4.6783876, -0.5129364, 0.5595711
1: -11.1307716, -9.8355656, -11.1173592, -9.8319168, -0.4915719, 0.4599959
2: 6.1268797, 7.3020010, 6.1202173, 7.2841597, -0.3909556, 0.4403259
3: -4.7563162, -3.9593551, -4.7660618, -3.9437408, -0.3639964, 0.3108318
4: -12.3542500, -11.2273092, -12.3431702, -11.2249250, -0.3911362, 0.3675801
5: -13.7801399, -12.7568789, -13.7816887, -12.7550278, -0.3525114, 0.3465185
6: -10.9240408, -9.7276697, -10.9256763, -9.7302485, -0.5096588, 0.5132845
7: -1.7039583, -0.7397051, -1.7077980, -0.7377782, -0.3258461, 0.3384167
8: -0.6198559, 0.2905149, -0.6199336, 0.2911777, -0.3437472, 0.3428868
9: -10.1057243, -8.8932705, -10.0893936, -8.8886776, -0.5136379, 0.4829150

Time for backsubstitution: 8.52 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1376

Time for candidate selection: 0.41 seconds

### Candidate
type: B, layer: 3, pos: 900

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3107802, upper bound: 0.3119842
time: 3.48 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3110358, upper bound: 0.3145957
time: 5.96 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.9714899, -4.6292105, -5.9904089, -4.6783876, -0.5222681, 0.6105230
1: -11.2169037, -9.8412638, -11.1173592, -9.8319168, -0.5929921, 0.4711105
2: 6.1509724, 7.3146191, 6.1202173, 7.2841597, -0.4030738, 0.4887000
3: -4.7271481, -3.9414175, -4.7660618, -3.9437408, -0.3691849, 0.3634567
4: -12.3540220, -11.2266998, -12.3431702, -11.2249250, -0.3911455, 0.3676943
5: -13.7808256, -12.7649765, -13.7816887, -12.7550278, -0.3599043, 0.3476359
6: -10.9242821, -9.7000914, -10.9256763, -9.7302485, -0.5212293, 0.5393456
7: -1.7305758, -0.7467840, -1.7077980, -0.7377782, -0.3590677, 0.3425910
8: -0.6065183, 0.3251286, -0.6199336, 0.2911777, -0.3631567, 0.3997304
9: -10.1078415, -8.8996964, -10.0893936, -8.8886776, -0.5191170, 0.4861803

Time for backsubstitution: 9.05 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1376

Time for candidate selection: 0.43 seconds

### Candidate
type: B, layer: 3, pos: 900

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3107802, upper bound: 0.3119842
time: 3.49 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3110358, upper bound: 0.3145959
time: 4.14 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.9622641, -4.6570978, -5.9659472, -4.6505003, -0.5607328, 0.5547476
1: -11.1151867, -9.8212585, -11.1976156, -9.8376789, -0.5004975, 0.5825443
2: 6.1426101, 7.3022623, 6.1478367, 7.2967577, -0.4458798, 0.4335099
3: -4.7475400, -3.9594254, -4.7356935, -3.9258316, -0.4223346, 0.3040242
4: -12.3180122, -11.2269630, -12.3280563, -11.2244091, -0.3522125, 0.3511679
5: -13.7801132, -12.7671642, -13.7821274, -12.7659817, -0.3432204, 0.3342993
6: -10.9297762, -9.7426710, -10.9251289, -9.7101841, -0.5418060, 0.5038714
7: -1.6851251, -0.7390087, -1.7271409, -0.7449319, -0.3181526, 0.3738860
8: -0.6213517, 0.2843571, -0.6064482, 0.3233757, -0.4183838, 0.3417507
9: -10.0890970, -8.8951302, -10.0851898, -8.8954439, -0.4953291, 0.4852444

Time for backsubstitution: 9.07 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1376

Time for candidate selection: 0.42 seconds

### Candidate
type: B, layer: 3, pos: 675

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3088032, upper bound: 0.3049696
time: 3.23 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3049702
time: 3.32 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.9818668, -4.6351581, -5.9742007, -4.6505003, -0.5802433, 0.5711243
1: -11.1274395, -9.8043118, -11.2022095, -9.8376350, -0.5057313, 0.5795796
2: 6.1368036, 7.3103380, 6.1472988, 7.2967710, -0.4554634, 0.4360985
3: -4.7472053, -3.9597671, -4.7356248, -3.9258120, -0.4221823, 0.3099811
4: -12.3507252, -11.1899357, -12.3415089, -11.2243586, -0.3683363, 0.4094873
5: -13.7816343, -12.7699356, -13.7823124, -12.7683992, -0.3784821, 0.3326444
6: -10.9656086, -9.7348137, -10.9257298, -9.7052670, -0.6012249, 0.5076591
7: -1.6977367, -0.7231772, -1.7321489, -0.7448726, -0.3244902, 0.3887228
8: -0.6337547, 0.2903128, -0.6065726, 0.3250685, -0.4387325, 0.3517430
9: -10.1012201, -8.8778915, -10.0898209, -8.8951883, -0.5000199, 0.5127853

Time for backsubstitution: 9.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1376

Time for candidate selection: 0.44 seconds

### Candidate
type: B, layer: 3, pos: 675

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3110359, upper bound: 0.3052266
time: 3.64 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3110360, upper bound: 0.3052261
time: 3.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 16.54 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.54
Output dim: 2, lower bound: -0.3074245, upper bound: 0.3132313
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.54
Output dim: 2, lower bound: -0.3074245, upper bound: 0.3166493
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.54
Output dim: 2, lower bound: -0.3074245, upper bound: 0.3132305
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.54
Output dim: 2, lower bound: -0.3074245, upper bound: 0.3166492
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.54
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3073636
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.54
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3107803
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.54
Output dim: 2, lower bound: -0.3052257, upper bound: 0.3076180
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.54
Output dim: 2, lower bound: -0.3052257, upper bound: 0.3110369
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.54
Output dim: 2, lower bound: -0.3107802, upper bound: 0.3119842
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.54
Output dim: 2, lower bound: -0.3110358, upper bound: 0.3145957
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.54
Output dim: 2, lower bound: -0.3107802, upper bound: 0.3119842
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.54
Output dim: 2, lower bound: -0.3110358, upper bound: 0.3145959
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.54
Output dim: 2, lower bound: -0.3088032, upper bound: 0.3049696
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.54
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3049702
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.54
Output dim: 2, lower bound: -0.3110359, upper bound: 0.3052266
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.54
Output dim: 2, lower bound: -0.3110360, upper bound: 0.3052261

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.9753256, -4.6783876, -5.9753256, -4.6783876, -0.5068521, 0.5068521
1: -11.1173582, -9.8418522, -11.1173582, -9.8418522, -0.4554036, 0.4554036
2: 6.1324129, 7.2841110, 6.1324129, 7.2841110, -0.3871943, 0.3871943
3: -4.7654700, -3.9379277, -4.7654700, -3.9379277, -0.3159095, 0.3159096
4: -12.3431702, -11.2331810, -12.3431702, -11.2331810, -0.3635247, 0.3635246
5: -13.7816658, -12.7541170, -13.7816658, -12.7541170, -0.3474994, 0.3474995
6: -10.9231339, -9.7315836, -10.9231339, -9.7315836, -0.5061769, 0.5061768
7: -1.7085752, -0.7377777, -1.7085752, -0.7377777, -0.3264912, 0.3264913
8: -0.6198463, 0.2913914, -0.6198463, 0.2913914, -0.3427488, 0.3427487
9: -10.0880108, -8.9000616, -10.0880108, -8.9000616, -0.4752263, 0.4752263

Time for backsubstitution: 9.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.41 seconds

### Candidate
type: A, layer: 3, pos: 900

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3061737, upper bound: 0.3109856
time: 3.54 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3087913, upper bound: 0.3111846
time: 3.61 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.9753256, -4.6783876, -5.9849749, -4.6570978, -0.5416656, 0.5231512
1: -11.1173582, -9.8418522, -11.1307716, -9.8355656, -0.4683187, 0.4780867
2: 6.1324129, 7.2841110, 6.1268797, 7.3020010, -0.4220585, 0.4051200
3: -4.7654700, -3.9379277, -4.7563162, -3.9593551, -0.3371465, 0.3427544
4: -12.3431702, -11.2331810, -12.3542500, -11.2273092, -0.3733821, 0.3807317
5: -13.7816658, -12.7541170, -13.7801399, -12.7568789, -0.3495773, 0.3504335
6: -10.9231339, -9.7315836, -10.9240408, -9.7276697, -0.5102737, 0.5076605
7: -1.7085752, -0.7377777, -1.7039583, -0.7397051, -0.3337028, 0.3305100
8: -0.6198463, 0.2913914, -0.6198559, 0.2905149, -0.3427807, 0.3433888
9: -10.0880108, -8.9000616, -10.1057243, -8.8932705, -0.4913305, 0.4974229

Time for backsubstitution: 8.39 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.41 seconds

### Candidate
type: A, layer: 3, pos: 900

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3061737, upper bound: 0.3144032
time: 3.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3087913, upper bound: 0.3146025
time: 3.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.9618578, -4.6505003, -5.9753256, -4.6783876, -0.5162003, 0.5582007
1: -11.2034893, -9.8475361, -11.1173582, -9.8418522, -0.5573211, 0.4665195
2: 6.1564984, 7.2967281, 6.1324129, 7.2841110, -0.3993124, 0.4358773
3: -4.7363181, -3.9199934, -4.7654700, -3.9379277, -0.3209751, 0.3685346
4: -12.3429394, -11.2325630, -12.3431702, -11.2331810, -0.3635347, 0.3636397
5: -13.7823524, -12.7622166, -13.7816658, -12.7541170, -0.3548955, 0.3486168
6: -10.9233694, -9.7040339, -10.9231339, -9.7315836, -0.5177455, 0.5321686
7: -1.7351809, -0.7448583, -1.7085752, -0.7377777, -0.3596824, 0.3306683
8: -0.6065111, 0.3260055, -0.6198463, 0.2913914, -0.3621446, 0.3995918
9: -10.0901184, -8.9064827, -10.0880108, -8.9000616, -0.4805166, 0.4784907

Time for backsubstitution: 9.05 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 900

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3109758
time: 3.38 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3052254, upper bound: 0.3111782
time: 3.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.9618578, -4.6505003, -5.9849749, -4.6570978, -0.5510137, 0.5744996
1: -11.2034893, -9.8475361, -11.1307716, -9.8355656, -0.5702362, 0.4892025
2: 6.1564984, 7.2967281, 6.1268797, 7.3020010, -0.4341764, 0.4538029
3: -4.7363181, -3.9199934, -4.7563162, -3.9593551, -0.3422120, 0.3953793
4: -12.3429394, -11.2325630, -12.3542500, -11.2273092, -0.3733923, 0.3808467
5: -13.7823524, -12.7622166, -13.7801399, -12.7568789, -0.3569734, 0.3515509
6: -10.9233694, -9.7040339, -10.9240408, -9.7276697, -0.5218422, 0.5336524
7: -1.7351809, -0.7448583, -1.7039583, -0.7397051, -0.3668940, 0.3346870
8: -0.6065111, 0.3260055, -0.6198559, 0.2905149, -0.3621767, 0.4002318
9: -10.0901184, -8.9064827, -10.1057243, -8.8932705, -0.4966208, 0.5006876

Time for backsubstitution: 9.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.43 seconds

### Candidate
type: A, layer: 3, pos: 900

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3143932
time: 3.36 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3052254, upper bound: 0.3145960
time: 4.28 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.9526472, -4.6783876, -5.9508638, -4.6505003, -0.5546737, 0.5022066
1: -11.1017761, -9.8275452, -11.1976147, -9.8476315, -0.4643502, 0.5779969
2: 6.1481256, 7.2843719, 6.1600294, 7.2967091, -0.4421316, 0.3804796
3: -4.7566972, -3.9379983, -4.7350860, -3.9200191, -0.3742123, 0.3091165
4: -12.3069334, -11.2328310, -12.3280573, -11.2326565, -0.3245957, 0.3471277
5: -13.7816381, -12.7643986, -13.7821007, -12.7650700, -0.3382203, 0.3352842
6: -10.9289207, -9.7465906, -10.9225826, -9.7115059, -0.5383532, 0.4967709
7: -1.6897473, -0.7370784, -1.7279162, -0.7449319, -0.3187994, 0.3619286
8: -0.6213489, 0.2852368, -0.6063595, 0.3235903, -0.4174185, 0.3416175
9: -10.0713844, -8.9019098, -10.0837889, -8.9068270, -0.4569758, 0.4775817

Time for backsubstitution: 9.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 172

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3073638
time: 3.17 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3073643
time: 3.45 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.9526472, -4.6783876, -5.9604883, -4.6292105, -0.5890903, 0.5181986
1: -11.1017761, -9.8275452, -11.2110291, -9.8413572, -0.4772620, 0.6001751
2: 6.1481256, 7.2843719, 6.1545086, 7.3146005, -0.4766885, 0.3982834
3: -4.7566972, -3.9379983, -4.7259154, -3.9414432, -0.3954537, 0.3360870
4: -12.3069334, -11.2328310, -12.3391380, -11.2267952, -0.3344759, 0.3643296
5: -13.7816381, -12.7643986, -13.7805767, -12.7678337, -0.3402982, 0.3382076
6: -10.9289207, -9.7465906, -10.9234962, -9.7075596, -0.5425146, 0.4982454
7: -1.6897473, -0.7370784, -1.7233121, -0.7468605, -0.3260062, 0.3661433
8: -0.6213489, 0.2852368, -0.6063643, 0.3227124, -0.4174513, 0.3422670
9: -10.0713844, -8.9019098, -10.1015158, -8.9000416, -0.4729893, 0.4999726

Time for backsubstitution: 9.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.41 seconds

### Candidate
type: A, layer: 3, pos: 172

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3107809
time: 3.37 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3107809
time: 3.43 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.9722652, -4.6564512, -5.9591222, -4.6505003, -0.5741827, 0.5185182
1: -11.1140299, -9.8105927, -11.2022123, -9.8475838, -0.4695734, 0.5750376
2: 6.1423125, 7.2924485, 6.1594920, 7.2967205, -0.4517149, 0.3830183
3: -4.7563605, -3.9383416, -4.7350159, -3.9200001, -0.3740408, 0.3150724
4: -12.3396454, -11.1958008, -12.3415108, -11.2326040, -0.3407139, 0.4054806
5: -13.7831650, -12.7671728, -13.7822876, -12.7674894, -0.3735588, 0.3336296
6: -10.9648972, -9.7387333, -10.9231892, -9.7065983, -0.5978136, 0.5005544
7: -1.7023544, -0.7212458, -1.7329237, -0.7448728, -0.3251369, 0.3768459
8: -0.6337538, 0.2911882, -0.6064825, 0.3252816, -0.4379207, 0.3516060
9: -10.0835028, -8.8846493, -10.0884132, -8.9065704, -0.4616011, 0.5051354

Time for backsubstitution: 9.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.49 seconds

### Candidate
type: A, layer: 3, pos: 172

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3052254, upper bound: 0.3076176
time: 3.75 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3052254, upper bound: 0.3076180
time: 3.64 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.9722652, -4.6564512, -5.9687490, -4.6292105, -0.6085994, 0.5346708
1: -11.1140299, -9.8105927, -11.2156219, -9.8413095, -0.4825001, 0.5972209
2: 6.1423125, 7.2924485, 6.1539655, 7.3146143, -0.4862716, 0.4009101
3: -4.7563605, -3.9383416, -4.7258430, -3.9414244, -0.3952910, 0.3420464
4: -12.3396454, -11.1958008, -12.3525925, -11.2267427, -0.3506016, 0.4226865
5: -13.7831650, -12.7671728, -13.7807636, -12.7702484, -0.3756366, 0.3365578
6: -10.9648972, -9.7387333, -10.9241009, -9.7026520, -0.6019878, 0.5020543
7: -1.7023544, -0.7212458, -1.7283175, -0.7468009, -0.3323449, 0.3810619
8: -0.6337538, 0.2911882, -0.6064868, 0.3244038, -0.4379535, 0.3522575
9: -10.0835028, -8.8846493, -10.1061420, -8.8997850, -0.4776808, 0.5275357

Time for backsubstitution: 9.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.44 seconds

### Candidate
type: A, layer: 3, pos: 172

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3052254, upper bound: 0.3110365
time: 3.56 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3052254, upper bound: 0.3110369
time: 3.53 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.9739647, -4.6570978, -5.9667015, -4.6783876, -0.5053807, 0.5448141
1: -11.1249065, -9.8356771, -11.1017752, -9.8322020, -0.4890151, 0.4542316
2: 6.1304579, 7.3019819, 6.1297021, 7.2841048, -0.3880830, 0.4335281
3: -4.7551479, -3.9593797, -4.7629757, -3.9438102, -0.3619964, 0.3060923
4: -12.3394003, -11.2274075, -12.3067837, -11.2251663, -0.3739557, 0.3281239
5: -13.7798834, -12.7597961, -13.7810030, -12.7628202, -0.3312076, 0.3378443
6: -10.9231548, -9.7342577, -10.9233036, -9.7450752, -0.4928584, 0.5046132
7: -1.6967440, -0.7397890, -1.6890440, -0.7380009, -0.3198562, 0.3275536
8: -0.6196837, 0.2881579, -0.6194839, 0.2850208, -0.3387607, 0.3401865
9: -10.0992260, -8.8936138, -10.0729465, -8.8895893, -0.5068784, 0.4660909

Time for backsubstitution: 9.19 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 410

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3089755, upper bound: 0.3059561
time: 3.31 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3081890, upper bound: 0.3062992
time: 3.64 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.9822569, -4.6570978, -5.9870009, -4.6564512, -0.5217320, 0.5641671
1: -11.1294794, -9.8356009, -11.1140308, -9.8154926, -0.4859202, 0.4600174
2: 6.1286764, 7.3019962, 6.1232896, 7.2921715, -0.3905970, 0.4430702
3: -4.7549462, -3.9593616, -4.7626643, -3.9441531, -0.3680140, 0.3059243
4: -12.3528099, -11.2273426, -12.3394985, -11.1881685, -0.4317605, 0.3442749
5: -13.7800770, -12.7618122, -13.7825031, -12.7654514, -0.3295323, 0.3730344
6: -10.9238310, -9.7302599, -10.9574757, -9.7371960, -0.4970355, 0.5622779
7: -1.7016480, -0.7397246, -1.7016306, -0.7224236, -0.3341701, 0.3333138
8: -0.6198244, 0.2897911, -0.6317539, 0.2909751, -0.3487070, 0.3629048
9: -10.1041212, -8.8933582, -10.0850964, -8.8722429, -0.5348256, 0.4707711

Time for backsubstitution: 9.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.46 seconds

### Candidate
type: A, layer: 3, pos: 410

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3101225, upper bound: 0.3077273
time: 3.95 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3081890, upper bound: 0.3081892
time: 3.57 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.9604883, -4.6292105, -5.9667015, -4.6783876, -0.5154626, 0.5957661
1: -11.2110291, -9.8413572, -11.1017752, -9.8322020, -0.5910959, 0.4653741
2: 6.1545086, 7.3146005, 6.1297021, 7.2841048, -0.4003338, 0.4818987
3: -4.7259154, -3.9414432, -4.7629757, -3.9438102, -0.3672187, 0.3587101
4: -12.3391380, -11.2267952, -12.3067837, -11.2251663, -0.3741162, 0.3282515
5: -13.7805767, -12.7678337, -13.7810030, -12.7628202, -0.3385680, 0.3387909
6: -10.9234962, -9.7075596, -10.9233036, -9.7450752, -0.5045471, 0.5301827
7: -1.7233121, -0.7468605, -1.6890440, -0.7380009, -0.3529375, 0.3317303
8: -0.6063643, 0.3227124, -0.6194839, 0.2850208, -0.3581984, 0.3961813
9: -10.1015158, -8.9000416, -10.0729465, -8.8895893, -0.5123613, 0.4693491

Time for backsubstitution: 9.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.41 seconds

### Candidate
type: A, layer: 3, pos: 410

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3036214, upper bound: 0.3059372
time: 4.04 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3049538, upper bound: 0.3062991
time: 3.57 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.9687490, -4.6292105, -5.9870009, -4.6564512, -0.5310789, 0.6151190
1: -11.2156219, -9.8413095, -11.1140308, -9.8154926, -0.5875523, 0.4711173
2: 6.1539655, 7.3146143, 6.1232896, 7.2921715, -0.4017917, 0.4914428
3: -4.7258430, -3.9414244, -4.7626643, -3.9441531, -0.3731712, 0.3585476
4: -12.3525925, -11.2267427, -12.3394985, -11.1881685, -0.4316339, 0.3443707
5: -13.7807636, -12.7702484, -13.7825031, -12.7654514, -0.3369183, 0.3740915
6: -10.9241009, -9.7026520, -10.9574757, -9.7371960, -0.5086386, 0.5893987
7: -1.7283175, -0.7468009, -1.7016306, -0.7224236, -0.3676062, 0.3374891
8: -0.6064868, 0.3244038, -0.6317539, 0.2909751, -0.3681141, 0.4198320
9: -10.1061420, -8.8997850, -10.0850964, -8.8722429, -0.5403240, 0.4740343

Time for backsubstitution: 9.21 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 900
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.49 seconds

### Candidate
type: A, layer: 3, pos: 410

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3051569, upper bound: 0.3077233
time: 3.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3049538, upper bound: 0.3081890
time: 3.88 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -5.9622641, -4.6570978, -5.9508638, -4.6505003, -0.5703537, 0.5370198
1: -11.1151867, -9.8212585, -11.1976147, -9.8476315, -0.4870112, 0.5904487
2: 6.1426101, 7.3022623, 6.1600294, 7.2967091, -0.4595457, 0.4153514
3: -4.7475400, -3.9594254, -4.7350860, -3.9200191, -0.4010871, 0.3303180
4: -12.3180122, -11.2269630, -12.3280573, -11.2326565, -0.3417883, 0.3568737
5: -13.7801132, -12.7671642, -13.7821007, -12.7650700, -0.3411425, 0.3373186
6: -10.9297762, -9.7426710, -10.9225826, -9.7115059, -0.5397732, 0.5008685
7: -1.6851251, -0.7390087, -1.7279162, -0.7449319, -0.3228173, 0.3691256
8: -0.6213517, 0.2843571, -0.6063595, 0.3235903, -0.4180253, 0.3416494
9: -10.0890970, -8.8951302, -10.0837889, -8.9068270, -0.4792054, 0.4933307

Time for backsubstitution: 9.27 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.44 seconds

### Candidate
type: A, layer: 3, pos: 172

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3088032, upper bound: 0.3049702
time: 3.47 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3088032, upper bound: 0.3049695
time: 5.03 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -5.9622641, -4.6570978, -5.9604883, -4.6292105, -0.5607328, 0.5082612
1: -11.1151867, -9.8212585, -11.2110291, -9.8413572, -0.4689320, 0.5825436
2: 6.1426101, 7.3022623, 6.1545086, 7.3146005, -0.4458599, 0.3842055
3: -4.7475400, -3.9594254, -4.7259154, -3.9414432, -0.3684759, 0.3033721
4: -12.3180122, -11.2269630, -12.3391380, -11.2267952, -0.3286486, 0.3511676
5: -13.7801132, -12.7671642, -13.7805767, -12.7678337, -0.3372226, 0.3342903
6: -10.9297762, -9.7426710, -10.9234962, -9.7075596, -0.5406334, 0.4990446
7: -1.6851251, -0.7390087, -1.7233121, -0.7468605, -0.3181571, 0.3613189
8: -0.6213517, 0.2843571, -0.6063643, 0.3227124, -0.4174170, 0.3416304
9: -10.0890970, -8.8951302, -10.1015158, -8.9000416, -0.4634975, 0.4841050

Time for backsubstitution: 9.18 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.47 seconds

### Candidate
type: A, layer: 3, pos: 172

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3049703
time: 3.49 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3049695
time: 4.87 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.9818668, -4.6351581, -5.9591222, -4.6505003, -0.5897114, 0.5533317
1: -11.1274395, -9.8043118, -11.2022123, -9.8475838, -0.4922310, 0.5870119
2: 6.1368036, 7.3103380, 6.1594920, 7.2967205, -0.4690754, 0.4178611
3: -4.7472053, -3.9597671, -4.7350159, -3.9200001, -0.4009258, 0.3362681
4: -12.3507252, -11.1899357, -12.3415108, -11.2326040, -0.3579043, 0.4148985
5: -13.7816343, -12.7699356, -13.7822876, -12.7674894, -0.3764043, 0.3356640
6: -10.9656086, -9.7348137, -10.9231892, -9.7065983, -0.5990274, 0.5046525
7: -1.6977367, -0.7231772, -1.7329237, -0.7448728, -0.3291547, 0.3839677
8: -0.6337547, 0.2903128, -0.6064825, 0.3252816, -0.4383736, 0.3516378
9: -10.1012201, -8.8778915, -10.0884132, -8.9065704, -0.4838305, 0.5206746

Time for backsubstitution: 9.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.41 seconds

### Candidate
type: A, layer: 3, pos: 172

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3110356, upper bound: 0.3052262
time: 4.64 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3110356, upper bound: 0.3052258
time: 3.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.9818668, -4.6351581, -5.9687490, -4.6292105, -0.5802433, 0.5245824
1: -11.1274395, -9.8043118, -11.2156219, -9.8413095, -0.4741609, 0.5795791
2: 6.1368036, 7.3103380, 6.1539655, 7.3146143, -0.4554431, 0.3867362
3: -4.7472053, -3.9597671, -4.7258430, -3.9414244, -0.3683017, 0.3093276
4: -12.3507252, -11.1899357, -12.3525925, -11.2267427, -0.3447677, 0.4094870
5: -13.7816343, -12.7699356, -13.7807636, -12.7702484, -0.3725651, 0.3326353
6: -10.9656086, -9.7348137, -10.9241009, -9.7026520, -0.6001163, 0.5028285
7: -1.6977367, -0.7231772, -1.7283175, -0.7468009, -0.3244947, 0.3762373
8: -0.6337547, 0.2903128, -0.6064868, 0.3244038, -0.4378929, 0.3516205
9: -10.1012201, -8.8778915, -10.1061420, -8.8997850, -0.4681217, 0.5116622

Time for backsubstitution: 9.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 172
type: A, layer: 3, pos: 410
type: A, layer: 3, pos: 1157
type: A, layer: 3, pos: 2363
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 2516
type: A, layer: 3, pos: 739
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 3125
type: A, layer: 3, pos: 425
type: A, layer: 3, pos: 614
type: A, layer: 3, pos: 2320
type: A, layer: 3, pos: 1849
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 1376

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 172

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3110357, upper bound: 0.3052262
time: 4.02 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3110357, upper bound: 0.3052258
time: 3.70 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 17.33 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3061737, upper bound: 0.3109856
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3087913, upper bound: 0.3111846
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3061737, upper bound: 0.3144032
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3087913, upper bound: 0.3146025
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3109758
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3052254, upper bound: 0.3111782
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3143932
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3052254, upper bound: 0.3145960
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3073638
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3073643
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3107809
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3029926, upper bound: 0.3107809
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3052254, upper bound: 0.3076176
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3052254, upper bound: 0.3076180
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3052254, upper bound: 0.3110365
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3052254, upper bound: 0.3110369
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3089755, upper bound: 0.3059561
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3081890, upper bound: 0.3062992
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3101225, upper bound: 0.3077273
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3081890, upper bound: 0.3081892
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3036214, upper bound: 0.3059372
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3049538, upper bound: 0.3062991
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3051569, upper bound: 0.3077233
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3049538, upper bound: 0.3081890
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3088032, upper bound: 0.3049702
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3088032, upper bound: 0.3049695
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3049703
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3088033, upper bound: 0.3049695
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3110356, upper bound: 0.3052262
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3110356, upper bound: 0.3052258
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3110357, upper bound: 0.3052262
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.33
Output dim: 2, lower bound: -0.3110357, upper bound: 0.3052258

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.9516091, -4.6783876, -5.9643250, -4.6783876, -0.4924273, 0.4993155
1: -11.1017752, -9.8421383, -11.1114941, -9.8419647, -0.4496450, 0.4528922
2: 6.1418958, 7.2840567, 6.1359777, 7.2840919, -0.3806862, 0.3843197
3: -4.7623873, -3.9379997, -4.7643046, -3.9379547, -0.3111811, 0.3139393
4: -12.3067837, -11.2334223, -12.3283234, -11.2332764, -0.3240701, 0.3463672
5: -13.7809801, -12.7619076, -13.7814102, -12.7570333, -0.3388255, 0.3262027
6: -10.9207563, -9.7464275, -10.9222488, -9.7381744, -0.4975243, 0.4893401
7: -1.6897936, -0.7380016, -1.7013597, -0.7378612, -0.3156228, 0.3205017
8: -0.6193972, 0.2852364, -0.6196761, 0.2890353, -0.3400595, 0.3377645
9: -10.0715466, -8.9009743, -10.0815086, -8.9004011, -0.4584049, 0.4686964

Time for backsubstitution: 9.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 410
type: B, layer: 3, pos: 900
type: B, layer: 3, pos: 1157
type: B, layer: 3, pos: 2363
type: B, layer: 3, pos: 1698
type: B, layer: 3, pos: 2516
type: B, layer: 3, pos: 739
type: B, layer: 3, pos: 2606
type: B, layer: 3, pos: 3125
type: B, layer: 3, pos: 425
type: B, layer: 3, pos: 614
type: B, layer: 3, pos: 2320
type: B, layer: 3, pos: 1849
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1376

Time for candidate selection: 0.43 seconds

### Candidate
type: B, layer: 3, pos: 410

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3025394, upper bound: 0.3055591
time: 3.35 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3028829, upper bound: 0.3047736
time: 3.37 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.9719148, -4.6564512, -5.9726090, -4.6783876, -0.5118821, 0.5156472
1: -11.1140308, -9.8254251, -11.1160660, -9.8418856, -0.4554254, 0.4502768
2: 6.1354866, 7.2921238, 6.1342039, 7.2841043, -0.3902768, 0.3868345
3: -4.7620735, -3.9383416, -4.7641001, -3.9379358, -0.3110049, 0.3199636
4: -12.3394947, -11.1964388, -12.3417320, -11.2332125, -0.3402196, 0.4044824
5: -13.7824802, -12.7645388, -13.7816010, -12.7590485, -0.3740202, 0.3245230
6: -10.9550905, -9.7385330, -10.9229259, -9.7341776, -0.5554273, 0.4935201
7: -1.7024088, -0.7224250, -1.7062669, -0.7377949, -0.3213833, 0.3348144
8: -0.6316695, 0.2911897, -0.6198173, 0.2906699, -0.3628279, 0.3477086
9: -10.0837078, -8.8835974, -10.0864067, -8.9001503, -0.4630816, 0.4968435

Time for backsubstitution: 9.05 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.37 + 544.98 = 602.35 seconds
