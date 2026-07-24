## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 4)
Time budget: 1800 seconds
Split limit: 100


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-37.5578117, -0.2815094, -37.5578117, -0.2815094, -37.2763023, 37.2763023)
1: (-17.6220856, 10.4812832, -17.6220856, 10.4812832, -28.1033688, 28.1033688)
2: (-14.4312439, 10.0883579, -14.4312439, 10.0883579, -24.5196018, 24.5196018)
3: (-14.9140787, 14.0766926, -14.9140787, 14.0766926, -28.9907722, 28.9907722)
4: (-15.1334782, 14.7552061, -15.1334782, 14.7552061, -29.8886833, 29.8886833)
5: (-14.2266178, 15.1941137, -14.2266178, 15.1941137, -29.4207306, 29.4207306)
6: (-20.8351784, 10.3227959, -20.8351784, 10.3227959, -31.1579742, 31.1579742)
7: (-17.3737793, 16.5286179, -17.3737793, 16.5286179, -33.5350494, 33.5350494)
8: (-16.3228550, 19.1706161, -16.3228550, 19.1706161, -35.4586983, 35.4586983)
9: (-15.1635761, 13.7782021, -15.1635761, 13.7782021, -28.7501259, 28.7501240)
10: (-23.5488243, 17.2546597, -23.5488243, 17.2546597, -40.8034821, 40.8034821)
11: (-26.2301464, 10.3602743, -26.2301464, 10.3602743, -36.5904198, 36.5904198)
12: (-24.2345219, 12.1104794, -24.2345219, 12.1104794, -36.3450012, 36.3450012)
13: (-22.1990852, 18.4545364, -22.1990852, 18.4545364, -40.6536217, 40.6536217)
14: (-47.8707161, -0.4218502, -47.8707161, -0.4218502, -47.2985992, 47.2985992)
15: (-19.6529541, 10.3302298, -19.6529541, 10.3302298, -29.9831848, 29.9831848)
16: (-24.9793587, 13.2617989, -24.9793587, 13.2617989, -37.7826004, 37.7826004)
17: (-43.9613876, 12.4236822, -43.9613876, 12.4236822, -55.0888214, 55.0888138)
18: (-20.4577484, 12.4874125, -20.4577484, 12.4874125, -32.9451599, 32.9451599)
19: (-17.9183464, 4.2855158, -17.9183464, 4.2855158, -22.2038612, 22.2038612)
20: (-15.2772102, 8.4781389, -15.2772102, 8.4781389, -23.7553482, 23.7553482)
21: (-25.8919182, 3.7857480, -25.8919182, 3.7857480, -29.6776657, 29.6776657)
22: (-32.9530487, -0.8713036, -32.9530487, -0.8713036, -30.7369690, 30.7369709)
23: (-17.9241028, 8.9438848, -17.9241028, 8.9438848, -26.8679886, 26.8679886)
24: (-25.2896576, 7.3427725, -25.2896576, 7.3427725, -31.1461220, 31.1461201)
25: (-18.3243713, 10.8286724, -18.3243713, 10.8286724, -29.1530437, 29.1530437)
26: (-23.7112923, 14.9050426, -23.7112923, 14.9050426, -38.6163330, 38.6163330)
27: (-26.2949791, 6.6883535, -26.2949791, 6.6883535, -31.9885292, 31.9885292)
28: (-17.3191872, 10.6421642, -17.3191872, 10.6421642, -27.7754974, 27.7754936)
29: (-40.1622734, -5.2830868, -40.1622734, -5.2830868, -33.9363327, 33.9363365)
30: (-20.8790817, 12.3533125, -20.8790817, 12.3533125, -33.2323952, 33.2323952)
31: (-23.7122688, 7.0026283, -23.7122688, 7.0026283, -30.7148972, 30.7148972)
32: (-27.6387272, 4.3835154, -27.6387272, 4.3835154, -31.1337280, 31.1337318)
33: (-30.5414295, 14.6008110, -30.5414295, 14.6008110, -44.2418518, 44.2418556)
34: (-25.9748363, 9.9279261, -25.9748363, 9.9279261, -35.9027634, 35.9027634)
35: (-27.7439537, 10.9692993, -27.7439537, 10.9692993, -38.3029709, 38.3029709)
36: (-27.1807117, 10.9039993, -27.1807117, 10.9039993, -37.6421356, 37.6421356)
37: (-37.2227592, 9.6494160, -37.2227592, 9.6494160, -45.5764465, 45.5764503)
38: (-29.7080078, 13.9970264, -29.7080078, 13.9970264, -43.7050323, 43.7050323)
39: (-38.4577255, 11.6211214, -38.4577255, 11.6211214, -49.4307251, 49.4307175)
40: (-30.4051437, 9.7967424, -30.4051437, 9.7967424, -38.5560684, 38.5560760)
41: (-22.3953133, 9.5435400, -22.3953133, 9.5435400, -31.9388542, 31.9388542)
42: (-16.3960018, 7.5732532, -16.3960018, 7.5732532, -23.6731377, 23.6731396)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.84 + 63.84 = 66.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -14.6721677, upper bound: 14.6721678

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 356
type: A, layer: 3, pos: 236
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 292
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 229
type: A, layer: 3, pos: 868
type: A, layer: 3, pos: 355
type: A, layer: 3, pos: 357
type: A, layer: 3, pos: 363
type: A, layer: 3, pos: 348
type: A, layer: 3, pos: 284
type: A, layer: 3, pos: 869
type: A, layer: 3, pos: 997
type: A, layer: 3, pos: 887
type: A, layer: 3, pos: 377
type: A, layer: 3, pos: 353
type: A, layer: 3, pos: 369
type: A, layer: 3, pos: 375
type: A, layer: 3, pos: 875
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 293
type: A, layer: 3, pos: 988
type: A, layer: 3, pos: 999
type: A, layer: 3, pos: 881
type: A, layer: 3, pos: 291
type: A, layer: 3, pos: 378
type: A, layer: 3, pos: 991
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 305
type: A, layer: 3, pos: 283
type: A, layer: 3, pos: 996
type: A, layer: 3, pos: 289
type: A, layer: 3, pos: 383
type: A, layer: 3, pos: 993
type: A, layer: 3, pos: 380
type: A, layer: 3, pos: 1009
type: A, layer: 3, pos: 893
type: A, layer: 3, pos: 877
type: A, layer: 3, pos: 331
type: A, layer: 3, pos: 361
type: A, layer: 3, pos: 339
type: A, layer: 3, pos: 849
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 338
type: A, layer: 3, pos: 311
type: A, layer: 3, pos: 1015
type: A, layer: 3, pos: 972
type: A, layer: 3, pos: 850
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 231
type: A, layer: 3, pos: 889
type: A, layer: 3, pos: 865
type: A, layer: 3, pos: 843
type: A, layer: 3, pos: 684
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 300
type: A, layer: 3, pos: 859
type: A, layer: 3, pos: 895
type: A, layer: 3, pos: 882
type: A, layer: 3, pos: 347
type: A, layer: 3, pos: 689
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 1023
type: A, layer: 3, pos: 379
type: A, layer: 3, pos: 841
type: A, layer: 3, pos: 382
type: A, layer: 3, pos: 644
type: A, layer: 3, pos: 860
type: A, layer: 3, pos: 695
type: A, layer: 3, pos: 223
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 346
type: A, layer: 3, pos: 329
type: A, layer: 3, pos: 306
type: A, layer: 3, pos: 724
type: A, layer: 3, pos: 1003
type: A, layer: 3, pos: 265
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 313
type: A, layer: 3, pos: 334
type: A, layer: 3, pos: 273
type: A, layer: 3, pos: 314
type: A, layer: 3, pos: 85
type: A, layer: 3, pos: 978
type: A, layer: 3, pos: 874
type: A, layer: 3, pos: 1005
type: A, layer: 3, pos: 58
type: A, layer: 3, pos: 1021
type: A, layer: 3, pos: 846
type: A, layer: 3, pos: 69
type: A, layer: 3, pos: 884
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 647
type: A, layer: 3, pos: 1017
type: A, layer: 3, pos: 699
type: A, layer: 3, pos: 274
type: A, layer: 3, pos: 977
type: A, layer: 3, pos: 299
type: A, layer: 3, pos: 894
type: A, layer: 3, pos: 974
type: A, layer: 3, pos: 995
type: A, layer: 3, pos: 370
type: A, layer: 3, pos: 851
type: A, layer: 3, pos: 646
type: A, layer: 3, pos: 698
type: A, layer: 3, pos: 876
type: A, layer: 3, pos: 667
type: A, layer: 3, pos: 260
type: A, layer: 3, pos: 381
type: A, layer: 3, pos: 1019
type: A, layer: 3, pos: 673
type: A, layer: 3, pos: 235
type: A, layer: 3, pos: 316
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 419
type: A, layer: 3, pos: 867
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 980
type: A, layer: 3, pos: 967
type: A, layer: 3, pos: 319
type: A, layer: 3, pos: 315
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 258
type: A, layer: 3, pos: 214
type: A, layer: 3, pos: 376
type: A, layer: 3, pos: 883
type: A, layer: 3, pos: 259
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 700
type: A, layer: 3, pos: 836
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 1020
type: A, layer: 3, pos: 842
type: A, layer: 3, pos: 61
type: A, layer: 3, pos: 1018
type: A, layer: 3, pos: 1014
type: A, layer: 3, pos: 688
type: A, layer: 3, pos: 336
type: A, layer: 3, pos: 56
type: A, layer: 3, pos: 272
type: A, layer: 3, pos: 1010
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 345
type: A, layer: 3, pos: 51
type: A, layer: 3, pos: 656
type: A, layer: 3, pos: 340
type: A, layer: 3, pos: 975
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 657
type: A, layer: 3, pos: 201
type: A, layer: 3, pos: 645
type: A, layer: 3, pos: 360
type: A, layer: 3, pos: 690
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 649
type: A, layer: 3, pos: 683
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 220
type: A, layer: 3, pos: 335
type: A, layer: 3, pos: 703
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 62
type: A, layer: 3, pos: 102
type: A, layer: 3, pos: 870
type: A, layer: 3, pos: 344
type: A, layer: 3, pos: 68
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 337
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 111
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 404
type: A, layer: 3, pos: 651
type: A, layer: 3, pos: 349
type: A, layer: 3, pos: 1013
type: A, layer: 3, pos: 858
type: A, layer: 3, pos: 981
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 242
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 861
type: A, layer: 3, pos: 1004
type: A, layer: 3, pos: 279
type: A, layer: 3, pos: 987
type: A, layer: 3, pos: 325
type: A, layer: 3, pos: 281
type: A, layer: 3, pos: 113
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 78
type: A, layer: 3, pos: 658
type: A, layer: 3, pos: 57
type: A, layer: 3, pos: 54
type: A, layer: 3, pos: 297
type: A, layer: 3, pos: 203
type: A, layer: 3, pos: 835
type: A, layer: 3, pos: 971
type: A, layer: 3, pos: 420
type: A, layer: 3, pos: 63
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 55
type: A, layer: 3, pos: 879
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 595
type: A, layer: 3, pos: 263
type: A, layer: 3, pos: 642
type: A, layer: 3, pos: 969
type: A, layer: 3, pos: 702
type: A, layer: 3, pos: 318
type: A, layer: 3, pos: 863
type: A, layer: 3, pos: 983
type: A, layer: 3, pos: 328
type: A, layer: 3, pos: 257
type: A, layer: 3, pos: 675
type: A, layer: 3, pos: 50
type: A, layer: 3, pos: 343
type: A, layer: 3, pos: 965
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 365
type: A, layer: 3, pos: 855
type: A, layer: 3, pos: 428
type: A, layer: 3, pos: 664
type: A, layer: 3, pos: 86
type: A, layer: 3, pos: 246
type: A, layer: 3, pos: 354
type: A, layer: 3, pos: 598
type: A, layer: 3, pos: 252
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 643
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 238
type: A, layer: 3, pos: 333
type: A, layer: 3, pos: 264
type: A, layer: 3, pos: 1012
type: A, layer: 3, pos: 124
type: A, layer: 3, pos: 982
type: A, layer: 3, pos: 262
type: A, layer: 3, pos: 648
type: A, layer: 3, pos: 641
type: A, layer: 3, pos: 985
type: A, layer: 3, pos: 857
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 77
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 372
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 666
type: A, layer: 3, pos: 322
type: A, layer: 3, pos: 84
type: A, layer: 3, pos: 873
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 665
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 324
type: A, layer: 3, pos: 109
type: A, layer: 3, pos: 844
type: A, layer: 3, pos: 82
type: A, layer: 3, pos: 589
type: A, layer: 3, pos: 663
type: A, layer: 3, pos: 1007
type: A, layer: 3, pos: 696
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 209
type: A, layer: 3, pos: 94
type: A, layer: 3, pos: 296
type: A, layer: 3, pos: 989
type: A, layer: 3, pos: 251
type: A, layer: 3, pos: 885
type: A, layer: 3, pos: 104
type: A, layer: 3, pos: 990
type: A, layer: 3, pos: 681
type: A, layer: 3, pos: 261
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 304
type: A, layer: 3, pos: 358
type: A, layer: 3, pos: 127
type: A, layer: 3, pos: 280
type: A, layer: 3, pos: 628
type: A, layer: 3, pos: 249
type: A, layer: 3, pos: 853
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 833
type: A, layer: 3, pos: 674
type: A, layer: 3, pos: 207
type: A, layer: 3, pos: 986
type: A, layer: 3, pos: 610
type: A, layer: 3, pos: 998
type: A, layer: 3, pos: 123
type: A, layer: 3, pos: 847
type: A, layer: 3, pos: 53
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 597
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 672
type: A, layer: 3, pos: 270
type: A, layer: 3, pos: 590
type: A, layer: 3, pos: 202
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 282
type: A, layer: 3, pos: 368
type: A, layer: 3, pos: 596
type: A, layer: 3, pos: 1001
type: A, layer: 3, pos: 321
type: A, layer: 3, pos: 364
type: A, layer: 3, pos: 834
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 126
type: A, layer: 3, pos: 970
type: A, layer: 3, pos: 362
type: A, layer: 3, pos: 973
type: A, layer: 3, pos: 275
type: A, layer: 3, pos: 1002
type: A, layer: 3, pos: 617
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 630
type: A, layer: 3, pos: 352
type: A, layer: 3, pos: 205
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 97
type: A, layer: 3, pos: 1006
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 276
type: A, layer: 3, pos: 52
type: A, layer: 3, pos: 606
type: A, layer: 3, pos: 112
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 602
type: A, layer: 3, pos: 845
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 588
type: A, layer: 3, pos: 119
type: A, layer: 3, pos: 320
type: A, layer: 3, pos: 215
type: A, layer: 3, pos: 217
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 593
type: A, layer: 3, pos: 871
type: A, layer: 3, pos: 979
type: A, layer: 3, pos: 594
type: A, layer: 3, pos: 629
type: A, layer: 3, pos: 301
type: A, layer: 3, pos: 1022
type: A, layer: 3, pos: 866
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 1008
type: A, layer: 3, pos: 89
type: A, layer: 3, pos: 114
type: A, layer: 3, pos: 87
type: A, layer: 3, pos: 862
type: A, layer: 3, pos: 587
type: A, layer: 3, pos: 580
type: A, layer: 3, pos: 88
type: A, layer: 3, pos: 966
type: A, layer: 3, pos: 852
type: A, layer: 3, pos: 210
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 960
type: A, layer: 3, pos: 694
type: A, layer: 3, pos: 692
type: A, layer: 3, pos: 371
type: A, layer: 3, pos: 1016
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 591
type: A, layer: 3, pos: 599
type: A, layer: 3, pos: 103
type: A, layer: 3, pos: 247
type: A, layer: 3, pos: 351
type: A, layer: 3, pos: 631
type: A, layer: 3, pos: 288
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 267
type: A, layer: 3, pos: 341
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 367
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 256
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 271
type: A, layer: 3, pos: 266
type: A, layer: 3, pos: 413
type: A, layer: 3, pos: 653
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 652
type: A, layer: 3, pos: 586
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 585
type: A, layer: 3, pos: 682
type: A, layer: 3, pos: 105
type: A, layer: 3, pos: 601
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 623
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 701
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 968
type: A, layer: 3, pos: 691
type: A, layer: 3, pos: 1011
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 685
type: A, layer: 3, pos: 125
type: A, layer: 3, pos: 639
type: A, layer: 3, pos: 650
type: A, layer: 3, pos: 74
type: A, layer: 3, pos: 687
type: A, layer: 3, pos: 680
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 269
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 303
type: A, layer: 3, pos: 618
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 622
type: A, layer: 3, pos: 285
type: A, layer: 3, pos: 609
type: A, layer: 3, pos: 636
type: A, layer: 3, pos: 405
type: A, layer: 3, pos: 607
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 243
type: A, layer: 3, pos: 626

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 356

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.2515759, upper bound: 14.6653834
time: 57.45 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.2515759, upper bound: 14.6653834
time: 60.19 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 117.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 117.89
Output dim: 4, lower bound: -14.2515759, upper bound: 14.6653834
IS_A2, status: Status.UNKNOWN, split count: 1, time: 117.89
Output dim: 4, lower bound: -14.2515759, upper bound: 14.6653834

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -37.4739952, -0.3004932, -37.5494194, -0.2834339, -37.1905594, 37.2489243
1: -17.5580101, 10.4648561, -17.6157093, 10.4796486, -28.0376587, 28.0805664
2: -14.2757969, 10.0701427, -14.4158411, 10.0865555, -24.3623524, 24.4859848
3: -14.7690849, 14.0477276, -14.8996830, 14.0737963, -28.8428802, 28.9474106
4: -14.9281874, 14.7359905, -15.1131487, 14.7532825, -29.6814690, 29.8491402
5: -14.0849934, 15.1739731, -14.2125826, 15.1921234, -29.2771168, 29.3865547
6: -20.7994633, 10.2554855, -20.8316097, 10.3160505, -31.1155128, 31.0870953
7: -17.2662201, 16.5131378, -17.3629074, 16.5270786, -33.4242325, 33.5089264
8: -16.1540527, 19.1408539, -16.3061104, 19.1676750, -35.2839203, 35.4123001
9: -15.1192532, 13.7080078, -15.1591711, 13.7711439, -28.6985397, 28.6767769
10: -23.4922657, 17.0078735, -23.5431767, 17.2302094, -40.7224731, 40.5510483
11: -26.1868706, 10.1109638, -26.2257824, 10.3352661, -36.5221367, 36.3367462
12: -24.2027283, 11.7994852, -24.2313766, 12.0797100, -36.2824402, 36.0308609
13: -22.1687183, 18.3957100, -22.1960583, 18.4486504, -40.6173706, 40.5917664
14: -47.8219643, -0.6076126, -47.8658752, -0.4402657, -47.2299423, 47.1085167
15: -19.5501804, 10.2969151, -19.6424561, 10.3268967, -29.8770771, 29.9393711
16: -24.9138050, 13.1436558, -24.9728203, 13.2498922, -37.7063446, 37.6541939
17: -43.9089127, 12.1483126, -43.9561768, 12.3963346, -55.0101471, 54.8079300
18: -20.4187355, 12.4415560, -20.4538784, 12.4827671, -32.9015045, 32.8954353
19: -17.8821640, 4.2019873, -17.9147530, 4.2772493, -22.1594124, 22.1167412
20: -15.2449818, 8.4316330, -15.2739887, 8.4735107, -23.7184925, 23.7056217
21: -25.8551636, 3.6564875, -25.8882713, 3.7729580, -29.6281223, 29.5447578
22: -32.9147339, -0.9469070, -32.9492455, -0.8788776, -30.6892548, 30.6342926
23: -17.8962727, 8.8600607, -17.9213428, 8.9354267, -26.8316994, 26.7814026
24: -25.2523575, 7.3256340, -25.2859459, 7.3410611, -31.1078873, 31.1254082
25: -18.2995110, 10.7651396, -18.3218975, 10.8222847, -29.1217957, 29.0870361
26: -23.6734314, 14.7520676, -23.7075233, 14.8898239, -38.5632553, 38.4595909
27: -26.2423801, 6.6745281, -26.2897377, 6.6869659, -31.9363174, 31.9690533
28: -17.2934952, 10.5998640, -17.3166389, 10.6379290, -27.7452354, 27.7281265
29: -40.1233482, -5.4482346, -40.1583939, -5.2995033, -33.8837700, 33.7593842
30: -20.8527565, 12.2628632, -20.8764324, 12.3443594, -33.1971169, 33.1392975
31: -23.6721535, 6.9580998, -23.7082787, 6.9981804, -30.6703339, 30.6663780
32: -27.6122303, 4.2924700, -27.6360741, 4.3745012, -31.0985870, 31.0404129
33: -30.4064388, 14.5593338, -30.5280495, 14.5966740, -44.0809021, 44.1856499
34: -25.9011497, 9.8880262, -25.9675179, 9.9239645, -35.8251152, 35.8555450
35: -27.6527424, 10.9415531, -27.7348747, 10.9665365, -38.1786423, 38.2650108
36: -27.1480255, 10.8774986, -27.1774330, 10.9013119, -37.6085434, 37.6088943
37: -37.1732521, 9.5992527, -37.2178268, 9.6443300, -45.5217590, 45.5124512
38: -29.6269455, 13.9693451, -29.6998825, 13.9942665, -43.6212120, 43.6692276
39: -38.3685722, 11.6025133, -38.4488029, 11.6192694, -49.3414383, 49.4029388
40: -30.3329468, 9.7854500, -30.3978901, 9.7956200, -38.4873123, 38.5378723
41: -22.3487549, 9.4839764, -22.3906803, 9.5375261, -31.8862801, 31.8746567
42: -16.3647785, 7.3994923, -16.3928909, 7.5560417, -23.6272621, 23.4927197

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 356
type: B, layer: 3, pos: 236
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 292
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 229
type: B, layer: 3, pos: 868
type: B, layer: 3, pos: 355
type: B, layer: 3, pos: 357
type: B, layer: 3, pos: 363
type: B, layer: 3, pos: 348
type: B, layer: 3, pos: 284
type: B, layer: 3, pos: 869
type: B, layer: 3, pos: 997
type: B, layer: 3, pos: 887
type: B, layer: 3, pos: 377
type: B, layer: 3, pos: 353
type: B, layer: 3, pos: 369
type: B, layer: 3, pos: 375
type: B, layer: 3, pos: 875
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 293
type: B, layer: 3, pos: 988
type: B, layer: 3, pos: 999
type: B, layer: 3, pos: 881
type: B, layer: 3, pos: 291
type: B, layer: 3, pos: 378
type: B, layer: 3, pos: 991
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 305
type: B, layer: 3, pos: 283
type: B, layer: 3, pos: 996
type: B, layer: 3, pos: 289
type: B, layer: 3, pos: 383
type: B, layer: 3, pos: 993
type: B, layer: 3, pos: 380
type: B, layer: 3, pos: 1009
type: B, layer: 3, pos: 893
type: B, layer: 3, pos: 877
type: B, layer: 3, pos: 331
type: B, layer: 3, pos: 361
type: B, layer: 3, pos: 339
type: B, layer: 3, pos: 849
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 338
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 1015
type: B, layer: 3, pos: 972
type: B, layer: 3, pos: 850
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 231
type: B, layer: 3, pos: 889
type: B, layer: 3, pos: 865
type: B, layer: 3, pos: 843
type: B, layer: 3, pos: 684
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 300
type: B, layer: 3, pos: 859
type: B, layer: 3, pos: 895
type: B, layer: 3, pos: 882
type: B, layer: 3, pos: 347
type: B, layer: 3, pos: 689
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 1023
type: B, layer: 3, pos: 379
type: B, layer: 3, pos: 841
type: B, layer: 3, pos: 382
type: B, layer: 3, pos: 644
type: B, layer: 3, pos: 860
type: B, layer: 3, pos: 695
type: B, layer: 3, pos: 223
type: B, layer: 3, pos: 239
type: B, layer: 3, pos: 346
type: B, layer: 3, pos: 329
type: B, layer: 3, pos: 306
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 1003
type: B, layer: 3, pos: 265
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 313
type: B, layer: 3, pos: 334
type: B, layer: 3, pos: 273
type: B, layer: 3, pos: 314
type: B, layer: 3, pos: 85
type: B, layer: 3, pos: 978
type: B, layer: 3, pos: 874
type: B, layer: 3, pos: 1005
type: B, layer: 3, pos: 58
type: B, layer: 3, pos: 1021
type: B, layer: 3, pos: 846
type: B, layer: 3, pos: 69
type: B, layer: 3, pos: 884
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 647
type: B, layer: 3, pos: 1017
type: B, layer: 3, pos: 699
type: B, layer: 3, pos: 274
type: B, layer: 3, pos: 977
type: B, layer: 3, pos: 299
type: B, layer: 3, pos: 894
type: B, layer: 3, pos: 974
type: B, layer: 3, pos: 995
type: B, layer: 3, pos: 370
type: B, layer: 3, pos: 851
type: B, layer: 3, pos: 646
type: B, layer: 3, pos: 698
type: B, layer: 3, pos: 876
type: B, layer: 3, pos: 667
type: B, layer: 3, pos: 260
type: B, layer: 3, pos: 381
type: B, layer: 3, pos: 1019
type: B, layer: 3, pos: 673
type: B, layer: 3, pos: 235
type: B, layer: 3, pos: 316
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 419
type: B, layer: 3, pos: 867
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 980
type: B, layer: 3, pos: 967
type: B, layer: 3, pos: 319
type: B, layer: 3, pos: 315
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 258
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 376
type: B, layer: 3, pos: 883
type: B, layer: 3, pos: 259
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 700
type: B, layer: 3, pos: 836
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 1020
type: B, layer: 3, pos: 842
type: B, layer: 3, pos: 61
type: B, layer: 3, pos: 1018
type: B, layer: 3, pos: 1014
type: B, layer: 3, pos: 688
type: B, layer: 3, pos: 336
type: B, layer: 3, pos: 56
type: B, layer: 3, pos: 272
type: B, layer: 3, pos: 1010
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 345
type: B, layer: 3, pos: 51
type: B, layer: 3, pos: 656
type: B, layer: 3, pos: 340
type: B, layer: 3, pos: 975
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 657
type: B, layer: 3, pos: 201
type: B, layer: 3, pos: 645
type: B, layer: 3, pos: 360
type: B, layer: 3, pos: 690
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 649
type: B, layer: 3, pos: 683
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 220
type: B, layer: 3, pos: 335
type: B, layer: 3, pos: 703
type: B, layer: 3, pos: 62
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 102
type: B, layer: 3, pos: 870
type: B, layer: 3, pos: 344
type: B, layer: 3, pos: 68
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 337
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 111
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 404
type: B, layer: 3, pos: 651
type: B, layer: 3, pos: 349
type: B, layer: 3, pos: 1013
type: B, layer: 3, pos: 858
type: B, layer: 3, pos: 981
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 242
type: B, layer: 3, pos: 861
type: B, layer: 3, pos: 1004
type: B, layer: 3, pos: 279
type: B, layer: 3, pos: 987
type: B, layer: 3, pos: 325
type: B, layer: 3, pos: 281
type: B, layer: 3, pos: 113
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 78
type: B, layer: 3, pos: 658
type: B, layer: 3, pos: 57
type: B, layer: 3, pos: 54
type: B, layer: 3, pos: 297
type: B, layer: 3, pos: 203
type: B, layer: 3, pos: 835
type: B, layer: 3, pos: 971
type: B, layer: 3, pos: 420
type: B, layer: 3, pos: 63
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 55
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 879
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 595
type: B, layer: 3, pos: 263
type: B, layer: 3, pos: 969
type: B, layer: 3, pos: 642
type: B, layer: 3, pos: 702
type: B, layer: 3, pos: 318
type: B, layer: 3, pos: 863
type: B, layer: 3, pos: 983
type: B, layer: 3, pos: 328
type: B, layer: 3, pos: 257
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 50
type: B, layer: 3, pos: 343
type: B, layer: 3, pos: 965
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 365
type: B, layer: 3, pos: 855
type: B, layer: 3, pos: 428
type: B, layer: 3, pos: 664
type: B, layer: 3, pos: 86
type: B, layer: 3, pos: 246
type: B, layer: 3, pos: 354
type: B, layer: 3, pos: 598
type: B, layer: 3, pos: 252
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 643
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 238
type: B, layer: 3, pos: 333
type: B, layer: 3, pos: 264
type: B, layer: 3, pos: 1012
type: B, layer: 3, pos: 124
type: B, layer: 3, pos: 982
type: B, layer: 3, pos: 262
type: B, layer: 3, pos: 648
type: B, layer: 3, pos: 641
type: B, layer: 3, pos: 985
type: B, layer: 3, pos: 857
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 77
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 372
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 666
type: B, layer: 3, pos: 322
type: B, layer: 3, pos: 84
type: B, layer: 3, pos: 873
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 81
type: B, layer: 3, pos: 665
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 324
type: B, layer: 3, pos: 109
type: B, layer: 3, pos: 844
type: B, layer: 3, pos: 82
type: B, layer: 3, pos: 589
type: B, layer: 3, pos: 1007
type: B, layer: 3, pos: 663
type: B, layer: 3, pos: 696
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 209
type: B, layer: 3, pos: 94
type: B, layer: 3, pos: 296
type: B, layer: 3, pos: 989
type: B, layer: 3, pos: 251
type: B, layer: 3, pos: 885
type: B, layer: 3, pos: 990
type: B, layer: 3, pos: 104
type: B, layer: 3, pos: 681
type: B, layer: 3, pos: 261
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 358
type: B, layer: 3, pos: 304
type: B, layer: 3, pos: 127
type: B, layer: 3, pos: 280
type: B, layer: 3, pos: 628
type: B, layer: 3, pos: 249
type: B, layer: 3, pos: 853
type: B, layer: 3, pos: 421
type: B, layer: 3, pos: 833
type: B, layer: 3, pos: 674
type: B, layer: 3, pos: 207
type: B, layer: 3, pos: 986
type: B, layer: 3, pos: 610
type: B, layer: 3, pos: 998
type: B, layer: 3, pos: 123
type: B, layer: 3, pos: 847
type: B, layer: 3, pos: 53
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 597
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 672
type: B, layer: 3, pos: 270
type: B, layer: 3, pos: 590
type: B, layer: 3, pos: 202
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 282
type: B, layer: 3, pos: 368
type: B, layer: 3, pos: 596
type: B, layer: 3, pos: 1001
type: B, layer: 3, pos: 321
type: B, layer: 3, pos: 364
type: B, layer: 3, pos: 834
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 126
type: B, layer: 3, pos: 970
type: B, layer: 3, pos: 362
type: B, layer: 3, pos: 973
type: B, layer: 3, pos: 275
type: B, layer: 3, pos: 1002
type: B, layer: 3, pos: 617
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 630
type: B, layer: 3, pos: 352
type: B, layer: 3, pos: 205
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 97
type: B, layer: 3, pos: 1006
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 276
type: B, layer: 3, pos: 52
type: B, layer: 3, pos: 606
type: B, layer: 3, pos: 112
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 602
type: B, layer: 3, pos: 845
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 588
type: B, layer: 3, pos: 119
type: B, layer: 3, pos: 320
type: B, layer: 3, pos: 215
type: B, layer: 3, pos: 217
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 593
type: B, layer: 3, pos: 871
type: B, layer: 3, pos: 979
type: B, layer: 3, pos: 594
type: B, layer: 3, pos: 629
type: B, layer: 3, pos: 301
type: B, layer: 3, pos: 1022
type: B, layer: 3, pos: 866
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1008
type: B, layer: 3, pos: 89
type: B, layer: 3, pos: 114
type: B, layer: 3, pos: 87
type: B, layer: 3, pos: 862
type: B, layer: 3, pos: 587
type: B, layer: 3, pos: 580
type: B, layer: 3, pos: 88
type: B, layer: 3, pos: 966
type: B, layer: 3, pos: 210
type: B, layer: 3, pos: 852
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 960
type: B, layer: 3, pos: 694
type: B, layer: 3, pos: 692
type: B, layer: 3, pos: 371
type: B, layer: 3, pos: 1016
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 591
type: B, layer: 3, pos: 599
type: B, layer: 3, pos: 103
type: B, layer: 3, pos: 247
type: B, layer: 3, pos: 351
type: B, layer: 3, pos: 288
type: B, layer: 3, pos: 631
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 267
type: B, layer: 3, pos: 341
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 367
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 256
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 271
type: B, layer: 3, pos: 266
type: B, layer: 3, pos: 413
type: B, layer: 3, pos: 653
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 586
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 585
type: B, layer: 3, pos: 682
type: B, layer: 3, pos: 105
type: B, layer: 3, pos: 601
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 623
type: B, layer: 3, pos: 110
type: B, layer: 3, pos: 701
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 968
type: B, layer: 3, pos: 691
type: B, layer: 3, pos: 1011
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 685
type: B, layer: 3, pos: 125
type: B, layer: 3, pos: 639
type: B, layer: 3, pos: 650
type: B, layer: 3, pos: 74
type: B, layer: 3, pos: 687
type: B, layer: 3, pos: 680
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 269
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 303
type: B, layer: 3, pos: 618
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 285
type: B, layer: 3, pos: 622
type: B, layer: 3, pos: 609
type: B, layer: 3, pos: 636
type: B, layer: 3, pos: 405
type: B, layer: 3, pos: 607
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 243
type: B, layer: 3, pos: 626

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 356

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.2515759, upper bound: 14.2515759
time: 65.57 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.2515759, upper bound: 14.6653834
time: 58.40 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -37.5874672, -0.1140747, -37.5536385, -0.2832489, -37.3042183, 37.4395638
1: -17.6272316, 10.6045780, -17.6158485, 10.4802799, -28.1075115, 28.2204266
2: -14.4560261, 10.4401169, -14.4291019, 10.0869217, -24.5429478, 24.8692188
3: -14.9321232, 14.3925228, -14.9111347, 14.0748081, -29.0069313, 29.3036575
4: -15.1902523, 15.2530766, -15.1293116, 14.7539091, -29.9441605, 30.3823891
5: -14.2574596, 15.5613327, -14.2231255, 15.1929283, -29.4503880, 29.7844582
6: -20.9744606, 10.2845631, -20.8318768, 10.2952795, -31.2697411, 31.1164398
7: -17.4000130, 16.7180061, -17.3626060, 16.5278511, -33.5522003, 33.7187691
8: -16.3712387, 19.5482483, -16.3198586, 19.1688576, -35.4995041, 35.8423500
9: -15.2238607, 13.7518034, -15.1606884, 13.7447262, -28.7736053, 28.7258797
10: -23.9916286, 17.2991066, -23.5455589, 17.2433777, -41.2350082, 40.8446655
11: -26.7594604, 10.3938522, -26.2255096, 10.3574228, -37.1168823, 36.6193619
12: -24.9852009, 12.1842766, -24.2333393, 12.1078053, -37.0930061, 36.4176178
13: -22.2565498, 18.4694691, -22.1959515, 18.4364796, -40.6930313, 40.6654205
14: -48.2951813, -0.3876228, -47.8680000, -0.4244347, -47.7178955, 47.3146591
15: -19.6210079, 10.4508438, -19.6126404, 10.3270006, -29.9480095, 30.0634842
16: -25.1297760, 13.1591167, -24.9746342, 13.2099123, -37.8879852, 37.6889191
17: -44.5407104, 12.4716606, -43.9595947, 12.4196587, -55.6644516, 55.1299591
18: -20.4693527, 12.5114365, -20.4304619, 12.4839039, -32.9532547, 32.9418983
19: -18.1600037, 4.2943230, -17.9164009, 4.2842937, -22.4442978, 22.2107239
20: -15.3805351, 8.4879808, -15.2749214, 8.4767590, -23.8572941, 23.7629013
21: -26.2186642, 3.8075819, -25.8895454, 3.7844253, -30.0030899, 29.6971283
22: -33.0367813, -0.8464785, -32.9513130, -0.8806992, -30.8622742, 30.7412376
23: -18.1613483, 8.9529963, -17.9229126, 8.9382725, -27.0996208, 26.8759079
24: -25.2795830, 7.3797107, -25.2711945, 7.3416653, -31.1474609, 31.1622314
25: -18.4259834, 10.8650064, -18.3222752, 10.8247175, -29.2507019, 29.1872826
26: -23.9901695, 14.9273062, -23.7090225, 14.8957796, -38.8859482, 38.6363297
27: -26.2956467, 6.7200375, -26.2740822, 6.6870375, -32.0245514, 31.9915619
28: -17.4510975, 10.6396008, -17.3180618, 10.6323729, -27.8987961, 27.7688942
29: -40.4591026, -5.2593994, -40.1605225, -5.2861118, -34.2611122, 33.9486313
30: -21.0336304, 12.3281021, -20.8757992, 12.3290730, -33.3627014, 33.2039032
31: -23.8607845, 7.0183582, -23.7099457, 7.0001159, -30.8609009, 30.7283039
32: -27.8206062, 4.3980007, -27.6363792, 4.3797665, -31.3196487, 31.1302681
33: -30.5883980, 14.8846264, -30.5396137, 14.5978584, -44.2445831, 44.5771637
34: -26.0082226, 10.0228977, -25.9707546, 9.9254951, -35.9337158, 35.9936523
35: -27.7753048, 11.1205292, -27.7393856, 10.9675179, -38.2881317, 38.5335541
36: -27.2416286, 10.9005461, -27.1754837, 10.8975124, -37.7013588, 37.6341743
37: -37.3477478, 9.6360989, -37.2194023, 9.6291599, -45.7160645, 45.5628281
38: -29.7592144, 14.0492496, -29.6967735, 13.9951458, -43.7543602, 43.7460251
39: -38.5343170, 11.7582817, -38.4486771, 11.6198139, -49.5032196, 49.5483856
40: -30.4489994, 9.8675900, -30.3981247, 9.7952652, -38.5910339, 38.6066284
41: -22.5260963, 9.5027838, -22.3922157, 9.5187712, -32.0448685, 31.8950005
42: -16.7768135, 7.6182723, -16.3943100, 7.5696821, -24.0625572, 23.6805954

Time for backsubstitution: 0.97 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 236
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 292
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 356
type: B, layer: 3, pos: 229
type: B, layer: 3, pos: 868
type: B, layer: 3, pos: 355
type: B, layer: 3, pos: 357
type: B, layer: 3, pos: 363
type: B, layer: 3, pos: 348
type: B, layer: 3, pos: 284
type: B, layer: 3, pos: 869
type: B, layer: 3, pos: 997
type: B, layer: 3, pos: 887
type: B, layer: 3, pos: 377
type: B, layer: 3, pos: 353
type: B, layer: 3, pos: 369
type: B, layer: 3, pos: 375
type: B, layer: 3, pos: 875
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 293
type: B, layer: 3, pos: 988
type: B, layer: 3, pos: 999
type: B, layer: 3, pos: 881
type: B, layer: 3, pos: 291
type: B, layer: 3, pos: 378
type: B, layer: 3, pos: 991
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 305
type: B, layer: 3, pos: 283
type: B, layer: 3, pos: 996
type: B, layer: 3, pos: 289
type: B, layer: 3, pos: 383
type: B, layer: 3, pos: 993
type: B, layer: 3, pos: 380
type: B, layer: 3, pos: 1009
type: B, layer: 3, pos: 893
type: B, layer: 3, pos: 877
type: B, layer: 3, pos: 331
type: B, layer: 3, pos: 361
type: B, layer: 3, pos: 339
type: B, layer: 3, pos: 849
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 338
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 1015
type: B, layer: 3, pos: 972
type: B, layer: 3, pos: 850
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 231
type: B, layer: 3, pos: 889
type: B, layer: 3, pos: 865
type: B, layer: 3, pos: 843
type: B, layer: 3, pos: 684
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 300
type: B, layer: 3, pos: 859
type: B, layer: 3, pos: 895
type: B, layer: 3, pos: 882
type: B, layer: 3, pos: 347
type: B, layer: 3, pos: 689
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 1023
type: B, layer: 3, pos: 379
type: B, layer: 3, pos: 841
type: B, layer: 3, pos: 382
type: B, layer: 3, pos: 644
type: B, layer: 3, pos: 860
type: B, layer: 3, pos: 695
type: B, layer: 3, pos: 223
type: B, layer: 3, pos: 239
type: B, layer: 3, pos: 346
type: B, layer: 3, pos: 329
type: B, layer: 3, pos: 306
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 1003
type: B, layer: 3, pos: 265
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 313
type: B, layer: 3, pos: 334
type: B, layer: 3, pos: 273
type: B, layer: 3, pos: 314
type: B, layer: 3, pos: 85
type: B, layer: 3, pos: 978
type: B, layer: 3, pos: 874
type: B, layer: 3, pos: 1005
type: B, layer: 3, pos: 58
type: B, layer: 3, pos: 1021
type: B, layer: 3, pos: 846
type: B, layer: 3, pos: 69
type: B, layer: 3, pos: 884
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 647
type: B, layer: 3, pos: 1017
type: B, layer: 3, pos: 699
type: B, layer: 3, pos: 274
type: B, layer: 3, pos: 977
type: B, layer: 3, pos: 299
type: B, layer: 3, pos: 894
type: B, layer: 3, pos: 974
type: B, layer: 3, pos: 995
type: B, layer: 3, pos: 370
type: B, layer: 3, pos: 851
type: B, layer: 3, pos: 646
type: B, layer: 3, pos: 698
type: B, layer: 3, pos: 876
type: B, layer: 3, pos: 667
type: B, layer: 3, pos: 260
type: B, layer: 3, pos: 381
type: B, layer: 3, pos: 1019
type: B, layer: 3, pos: 673
type: B, layer: 3, pos: 235
type: B, layer: 3, pos: 316
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 419
type: B, layer: 3, pos: 867
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 980
type: B, layer: 3, pos: 967
type: B, layer: 3, pos: 319
type: B, layer: 3, pos: 315
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 258
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 376
type: B, layer: 3, pos: 883
type: B, layer: 3, pos: 259
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 700
type: B, layer: 3, pos: 836
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 1020
type: B, layer: 3, pos: 842
type: B, layer: 3, pos: 61
type: B, layer: 3, pos: 1018
type: B, layer: 3, pos: 1014
type: B, layer: 3, pos: 688
type: B, layer: 3, pos: 336
type: B, layer: 3, pos: 56
type: B, layer: 3, pos: 272
type: B, layer: 3, pos: 1010
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 345
type: B, layer: 3, pos: 51
type: B, layer: 3, pos: 656
type: B, layer: 3, pos: 340
type: B, layer: 3, pos: 975
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 657
type: B, layer: 3, pos: 201
type: B, layer: 3, pos: 645
type: B, layer: 3, pos: 360
type: B, layer: 3, pos: 690
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 649
type: B, layer: 3, pos: 683
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 220
type: B, layer: 3, pos: 335
type: B, layer: 3, pos: 703
type: B, layer: 3, pos: 62
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 102
type: B, layer: 3, pos: 870
type: B, layer: 3, pos: 344
type: B, layer: 3, pos: 68
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 337
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 111
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 404
type: B, layer: 3, pos: 651
type: B, layer: 3, pos: 349
type: B, layer: 3, pos: 1013
type: B, layer: 3, pos: 858
type: B, layer: 3, pos: 981
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 242
type: B, layer: 3, pos: 861
type: B, layer: 3, pos: 1004
type: B, layer: 3, pos: 279
type: B, layer: 3, pos: 987
type: B, layer: 3, pos: 325
type: B, layer: 3, pos: 281
type: B, layer: 3, pos: 113
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 78
type: B, layer: 3, pos: 658
type: B, layer: 3, pos: 57
type: B, layer: 3, pos: 54
type: B, layer: 3, pos: 297
type: B, layer: 3, pos: 203
type: B, layer: 3, pos: 835
type: B, layer: 3, pos: 971
type: B, layer: 3, pos: 420
type: B, layer: 3, pos: 63
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 55
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 879
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 595
type: B, layer: 3, pos: 263
type: B, layer: 3, pos: 969
type: B, layer: 3, pos: 642
type: B, layer: 3, pos: 702
type: B, layer: 3, pos: 318
type: B, layer: 3, pos: 863
type: B, layer: 3, pos: 983
type: B, layer: 3, pos: 328
type: B, layer: 3, pos: 257
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 50
type: B, layer: 3, pos: 343
type: B, layer: 3, pos: 965
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 365
type: B, layer: 3, pos: 855
type: B, layer: 3, pos: 428
type: B, layer: 3, pos: 664
type: B, layer: 3, pos: 86
type: B, layer: 3, pos: 246
type: B, layer: 3, pos: 354
type: B, layer: 3, pos: 598
type: B, layer: 3, pos: 252
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 643
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 238
type: B, layer: 3, pos: 333
type: B, layer: 3, pos: 264
type: B, layer: 3, pos: 1012
type: B, layer: 3, pos: 124
type: B, layer: 3, pos: 982
type: B, layer: 3, pos: 262
type: B, layer: 3, pos: 648
type: B, layer: 3, pos: 641
type: B, layer: 3, pos: 985
type: B, layer: 3, pos: 857
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 77
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 372
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 666
type: B, layer: 3, pos: 322
type: B, layer: 3, pos: 84
type: B, layer: 3, pos: 873
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 81
type: B, layer: 3, pos: 665
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 324
type: B, layer: 3, pos: 109
type: B, layer: 3, pos: 844
type: B, layer: 3, pos: 82
type: B, layer: 3, pos: 589
type: B, layer: 3, pos: 663
type: B, layer: 3, pos: 1007
type: B, layer: 3, pos: 696
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 209
type: B, layer: 3, pos: 94
type: B, layer: 3, pos: 296
type: B, layer: 3, pos: 989
type: B, layer: 3, pos: 251
type: B, layer: 3, pos: 885
type: B, layer: 3, pos: 990
type: B, layer: 3, pos: 104
type: B, layer: 3, pos: 681
type: B, layer: 3, pos: 261
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 304
type: B, layer: 3, pos: 358
type: B, layer: 3, pos: 127
type: B, layer: 3, pos: 280
type: B, layer: 3, pos: 628
type: B, layer: 3, pos: 249
type: B, layer: 3, pos: 853
type: B, layer: 3, pos: 421
type: B, layer: 3, pos: 833
type: B, layer: 3, pos: 674
type: B, layer: 3, pos: 207
type: B, layer: 3, pos: 986
type: B, layer: 3, pos: 610
type: B, layer: 3, pos: 998
type: B, layer: 3, pos: 123
type: B, layer: 3, pos: 847
type: B, layer: 3, pos: 53
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 597
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 672
type: B, layer: 3, pos: 270
type: B, layer: 3, pos: 590
type: B, layer: 3, pos: 202
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 282
type: B, layer: 3, pos: 368
type: B, layer: 3, pos: 596
type: B, layer: 3, pos: 1001
type: B, layer: 3, pos: 321
type: B, layer: 3, pos: 364
type: B, layer: 3, pos: 834
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 126
type: B, layer: 3, pos: 970
type: B, layer: 3, pos: 362
type: B, layer: 3, pos: 973
type: B, layer: 3, pos: 275
type: B, layer: 3, pos: 1002
type: B, layer: 3, pos: 617
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 630
type: B, layer: 3, pos: 352
type: B, layer: 3, pos: 205
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 97
type: B, layer: 3, pos: 1006
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 276
type: B, layer: 3, pos: 52
type: B, layer: 3, pos: 606
type: B, layer: 3, pos: 112
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 602
type: B, layer: 3, pos: 845
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 588
type: B, layer: 3, pos: 119
type: B, layer: 3, pos: 320
type: B, layer: 3, pos: 215
type: B, layer: 3, pos: 217
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 593
type: B, layer: 3, pos: 871
type: B, layer: 3, pos: 979
type: B, layer: 3, pos: 594
type: B, layer: 3, pos: 629
type: B, layer: 3, pos: 301
type: B, layer: 3, pos: 1022
type: B, layer: 3, pos: 866
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1008
type: B, layer: 3, pos: 89
type: B, layer: 3, pos: 114
type: B, layer: 3, pos: 87
type: B, layer: 3, pos: 862
type: B, layer: 3, pos: 587
type: B, layer: 3, pos: 580
type: B, layer: 3, pos: 88
type: B, layer: 3, pos: 966
type: B, layer: 3, pos: 210
type: B, layer: 3, pos: 852
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 960
type: B, layer: 3, pos: 694
type: B, layer: 3, pos: 692
type: B, layer: 3, pos: 371
type: B, layer: 3, pos: 1016
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 591
type: B, layer: 3, pos: 599
type: B, layer: 3, pos: 103
type: B, layer: 3, pos: 247
type: B, layer: 3, pos: 351
type: B, layer: 3, pos: 288
type: B, layer: 3, pos: 631
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 267
type: B, layer: 3, pos: 341
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 367
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 256
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 271
type: B, layer: 3, pos: 266
type: B, layer: 3, pos: 413
type: B, layer: 3, pos: 653
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 586
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 585
type: B, layer: 3, pos: 682
type: B, layer: 3, pos: 105
type: B, layer: 3, pos: 601
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 623
type: B, layer: 3, pos: 110
type: B, layer: 3, pos: 701
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 968
type: B, layer: 3, pos: 691
type: B, layer: 3, pos: 1011
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 685
type: B, layer: 3, pos: 125
type: B, layer: 3, pos: 639
type: B, layer: 3, pos: 650
type: B, layer: 3, pos: 74
type: B, layer: 3, pos: 687
type: B, layer: 3, pos: 680
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 269
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 303
type: B, layer: 3, pos: 618
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 285
type: B, layer: 3, pos: 622
type: B, layer: 3, pos: 609
type: B, layer: 3, pos: 636
type: B, layer: 3, pos: 405
type: B, layer: 3, pos: 607
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 243
type: B, layer: 3, pos: 626

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 236

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.2253864, upper bound: 14.5280085
time: 49.21 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.2253864, upper bound: 14.6499223
time: 61.94 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 112.34 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 112.34
Output dim: 4, lower bound: -14.2515759, upper bound: 14.2515759
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 112.34
Output dim: 4, lower bound: -14.2515759, upper bound: 14.6653834
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 112.34
Output dim: 4, lower bound: -14.2253864, upper bound: 14.5280085
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 112.34
Output dim: 4, lower bound: -14.2253864, upper bound: 14.6499223

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -37.4739952, -0.3004932, -37.4739952, -0.3004932, -37.1735001, 37.1735001
1: -17.5580101, 10.4648561, -17.5580101, 10.4648561, -28.0228653, 28.0228653
2: -14.2757969, 10.0701427, -14.2757969, 10.0701427, -24.3459396, 24.3459396
3: -14.7690849, 14.0477276, -14.7690849, 14.0477276, -28.8168125, 28.8168125
4: -14.9281874, 14.7359905, -14.9281874, 14.7359905, -29.6641769, 29.6641769
5: -14.0849934, 15.1739731, -14.0849934, 15.1739731, -29.2589664, 29.2589664
6: -20.7994633, 10.2554855, -20.7994633, 10.2554855, -31.0549488, 31.0549488
7: -17.2662201, 16.5131378, -17.2662201, 16.5131378, -33.4105072, 33.4105072
8: -16.1540527, 19.1408539, -16.1540527, 19.1408539, -35.2574654, 35.2574577
9: -15.1192532, 13.7080078, -15.1192532, 13.7080078, -28.6367111, 28.6367111
10: -23.4922657, 17.0078735, -23.4922657, 17.0078735, -40.5001373, 40.5001373
11: -26.1868706, 10.1109638, -26.1868706, 10.1109638, -36.2978363, 36.2978363
12: -24.2027283, 11.7994852, -24.2027283, 11.7994852, -36.0022125, 36.0022125
13: -22.1687183, 18.3957100, -22.1687183, 18.3957100, -40.5644302, 40.5644302
14: -47.8219643, -0.6076126, -47.8219643, -0.6076126, -47.0631943, 47.0631905
15: -19.5501804, 10.2969151, -19.5501804, 10.2969151, -29.8470955, 29.8470955
16: -24.9138050, 13.1436558, -24.9138050, 13.1436558, -37.5965614, 37.5965652
17: -43.9089127, 12.1483126, -43.9089127, 12.1483126, -54.7617416, 54.7617340
18: -20.4187355, 12.4415560, -20.4187355, 12.4415560, -32.8602905, 32.8602905
19: -17.8821640, 4.2019873, -17.8821640, 4.2019873, -22.0841522, 22.0841522
20: -15.2449818, 8.4316330, -15.2449818, 8.4316330, -23.6766148, 23.6766148
21: -25.8551636, 3.6564875, -25.8551636, 3.6564875, -29.5116501, 29.5116501
22: -32.9147339, -0.9469070, -32.9147339, -0.9469070, -30.6002274, 30.6002274
23: -17.8962727, 8.8600607, -17.8962727, 8.8600607, -26.7563324, 26.7563324
24: -25.2523575, 7.3256340, -25.2523575, 7.3256340, -31.0924988, 31.0924988
25: -18.2995110, 10.7651396, -18.2995110, 10.7651396, -29.0646515, 29.0646515
26: -23.6734314, 14.7520676, -23.6734314, 14.7520676, -38.4254990, 38.4254990
27: -26.2423801, 6.6745281, -26.2423801, 6.6745281, -31.9233780, 31.9233761
28: -17.2934952, 10.5998640, -17.2934952, 10.5998640, -27.7049179, 27.7049179
29: -40.1233482, -5.4482346, -40.1233482, -5.4482346, -33.7275162, 33.7275162
30: -20.8527565, 12.2628632, -20.8527565, 12.2628632, -33.1156197, 33.1156197
31: -23.6721535, 6.9580998, -23.6721535, 6.9580998, -30.6302528, 30.6302528
32: -27.6122303, 4.2924700, -27.6122303, 4.2924700, -31.0169296, 31.0169277
33: -30.4064388, 14.5593338, -30.4064388, 14.5593338, -44.0442886, 44.0442924
34: -25.9011497, 9.8880262, -25.9011497, 9.8880262, -35.7891769, 35.7891769
35: -27.6527424, 10.9415531, -27.6527424, 10.9415531, -38.1553078, 38.1553078
36: -27.1480255, 10.8774986, -27.1480255, 10.8774986, -37.5813980, 37.5814018
37: -37.1732521, 9.5992527, -37.1732521, 9.5992527, -45.4685059, 45.4685020
38: -29.6269455, 13.9693451, -29.6269455, 13.9693451, -43.5962906, 43.5962906
39: -38.3685722, 11.6025133, -38.3685722, 11.6025133, -49.3242950, 49.3242912
40: -30.3329468, 9.7854500, -30.3329468, 9.7854500, -38.4770126, 38.4770126
41: -22.3487549, 9.4839764, -22.3487549, 9.4839764, -31.8327312, 31.8327312
42: -16.3647785, 7.3994923, -16.3647785, 7.3994923, -23.4672604, 23.4672585

Time for backsubstitution: 0.94 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 236
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 292
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 229
type: A, layer: 3, pos: 868
type: A, layer: 3, pos: 355
type: A, layer: 3, pos: 357
type: A, layer: 3, pos: 363
type: A, layer: 3, pos: 348
type: A, layer: 3, pos: 284
type: A, layer: 3, pos: 869
type: A, layer: 3, pos: 997
type: A, layer: 3, pos: 887
type: A, layer: 3, pos: 377
type: A, layer: 3, pos: 353
type: A, layer: 3, pos: 369
type: A, layer: 3, pos: 375
type: A, layer: 3, pos: 875
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 293
type: A, layer: 3, pos: 988
type: A, layer: 3, pos: 999
type: A, layer: 3, pos: 881
type: A, layer: 3, pos: 291
type: A, layer: 3, pos: 378
type: A, layer: 3, pos: 991
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 283
type: A, layer: 3, pos: 305
type: A, layer: 3, pos: 996
type: A, layer: 3, pos: 289
type: A, layer: 3, pos: 383
type: A, layer: 3, pos: 993
type: A, layer: 3, pos: 380
type: A, layer: 3, pos: 1009
type: A, layer: 3, pos: 893
type: A, layer: 3, pos: 877
type: A, layer: 3, pos: 331
type: A, layer: 3, pos: 361
type: A, layer: 3, pos: 339
type: A, layer: 3, pos: 849
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 338
type: A, layer: 3, pos: 311
type: A, layer: 3, pos: 1015
type: A, layer: 3, pos: 972
type: A, layer: 3, pos: 850
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 231
type: A, layer: 3, pos: 889
type: A, layer: 3, pos: 865
type: A, layer: 3, pos: 843
type: A, layer: 3, pos: 684
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 300
type: A, layer: 3, pos: 859
type: A, layer: 3, pos: 895
type: A, layer: 3, pos: 882
type: A, layer: 3, pos: 347
type: A, layer: 3, pos: 689
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 841
type: A, layer: 3, pos: 1023
type: A, layer: 3, pos: 379
type: A, layer: 3, pos: 382
type: A, layer: 3, pos: 644
type: A, layer: 3, pos: 860
type: A, layer: 3, pos: 695
type: A, layer: 3, pos: 223
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 346
type: A, layer: 3, pos: 329
type: A, layer: 3, pos: 306
type: A, layer: 3, pos: 265
type: A, layer: 3, pos: 724
type: A, layer: 3, pos: 1003
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 313
type: A, layer: 3, pos: 273
type: A, layer: 3, pos: 334
type: A, layer: 3, pos: 314
type: A, layer: 3, pos: 85
type: A, layer: 3, pos: 978
type: A, layer: 3, pos: 874
type: A, layer: 3, pos: 1005
type: A, layer: 3, pos: 58
type: A, layer: 3, pos: 1021
type: A, layer: 3, pos: 846
type: A, layer: 3, pos: 69
type: A, layer: 3, pos: 884
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 647
type: A, layer: 3, pos: 1017
type: A, layer: 3, pos: 699
type: A, layer: 3, pos: 274
type: A, layer: 3, pos: 977
type: A, layer: 3, pos: 299
type: A, layer: 3, pos: 894
type: A, layer: 3, pos: 974
type: A, layer: 3, pos: 995
type: A, layer: 3, pos: 370
type: A, layer: 3, pos: 851
type: A, layer: 3, pos: 646
type: A, layer: 3, pos: 698
type: A, layer: 3, pos: 876
type: A, layer: 3, pos: 667
type: A, layer: 3, pos: 260
type: A, layer: 3, pos: 381
type: A, layer: 3, pos: 235
type: A, layer: 3, pos: 673
type: A, layer: 3, pos: 1019
type: A, layer: 3, pos: 316
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 419
type: A, layer: 3, pos: 867
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 980
type: A, layer: 3, pos: 967
type: A, layer: 3, pos: 319
type: A, layer: 3, pos: 315
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 258
type: A, layer: 3, pos: 214
type: A, layer: 3, pos: 376
type: A, layer: 3, pos: 259
type: A, layer: 3, pos: 883
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 700
type: A, layer: 3, pos: 836
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 1020
type: A, layer: 3, pos: 842
type: A, layer: 3, pos: 61
type: A, layer: 3, pos: 1018
type: A, layer: 3, pos: 1014
type: A, layer: 3, pos: 336
type: A, layer: 3, pos: 688
type: A, layer: 3, pos: 56
type: A, layer: 3, pos: 272
type: A, layer: 3, pos: 1010
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 656
type: A, layer: 3, pos: 51
type: A, layer: 3, pos: 345
type: A, layer: 3, pos: 340
type: A, layer: 3, pos: 975
type: A, layer: 3, pos: 657
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 201
type: A, layer: 3, pos: 645
type: A, layer: 3, pos: 360
type: A, layer: 3, pos: 690
type: A, layer: 3, pos: 649
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 683
type: A, layer: 3, pos: 220
type: A, layer: 3, pos: 335
type: A, layer: 3, pos: 62
type: A, layer: 3, pos: 703
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 102
type: A, layer: 3, pos: 870
type: A, layer: 3, pos: 344
type: A, layer: 3, pos: 68
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 337
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 111
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 404
type: A, layer: 3, pos: 651
type: A, layer: 3, pos: 1013
type: A, layer: 3, pos: 349
type: A, layer: 3, pos: 858
type: A, layer: 3, pos: 981
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 242
type: A, layer: 3, pos: 861
type: A, layer: 3, pos: 1004
type: A, layer: 3, pos: 987
type: A, layer: 3, pos: 279
type: A, layer: 3, pos: 325
type: A, layer: 3, pos: 113
type: A, layer: 3, pos: 281
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 78
type: A, layer: 3, pos: 658
type: A, layer: 3, pos: 57
type: A, layer: 3, pos: 54
type: A, layer: 3, pos: 297
type: A, layer: 3, pos: 203
type: A, layer: 3, pos: 835
type: A, layer: 3, pos: 971
type: A, layer: 3, pos: 420
type: A, layer: 3, pos: 63
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 55
type: A, layer: 3, pos: 879
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 595
type: A, layer: 3, pos: 969
type: A, layer: 3, pos: 263
type: A, layer: 3, pos: 642
type: A, layer: 3, pos: 702
type: A, layer: 3, pos: 318
type: A, layer: 3, pos: 863
type: A, layer: 3, pos: 983
type: A, layer: 3, pos: 328
type: A, layer: 3, pos: 257
type: A, layer: 3, pos: 675
type: A, layer: 3, pos: 50
type: A, layer: 3, pos: 343
type: A, layer: 3, pos: 965
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 365
type: A, layer: 3, pos: 855
type: A, layer: 3, pos: 428
type: A, layer: 3, pos: 664
type: A, layer: 3, pos: 86
type: A, layer: 3, pos: 246
type: A, layer: 3, pos: 354
type: A, layer: 3, pos: 252
type: A, layer: 3, pos: 598
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 643
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 238
type: A, layer: 3, pos: 333
type: A, layer: 3, pos: 1012
type: A, layer: 3, pos: 264
type: A, layer: 3, pos: 982
type: A, layer: 3, pos: 124
type: A, layer: 3, pos: 262
type: A, layer: 3, pos: 648
type: A, layer: 3, pos: 641
type: A, layer: 3, pos: 985
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 77
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 857
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 372
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 666
type: A, layer: 3, pos: 322
type: A, layer: 3, pos: 84
type: A, layer: 3, pos: 873
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 665
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 324
type: A, layer: 3, pos: 109
type: A, layer: 3, pos: 844
type: A, layer: 3, pos: 1007
type: A, layer: 3, pos: 82
type: A, layer: 3, pos: 589
type: A, layer: 3, pos: 663
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 696
type: A, layer: 3, pos: 209
type: A, layer: 3, pos: 94
type: A, layer: 3, pos: 296
type: A, layer: 3, pos: 989
type: A, layer: 3, pos: 885
type: A, layer: 3, pos: 251
type: A, layer: 3, pos: 990
type: A, layer: 3, pos: 104
type: A, layer: 3, pos: 681
type: A, layer: 3, pos: 261
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 358
type: A, layer: 3, pos: 127
type: A, layer: 3, pos: 304
type: A, layer: 3, pos: 280
type: A, layer: 3, pos: 628
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 853
type: A, layer: 3, pos: 249
type: A, layer: 3, pos: 833
type: A, layer: 3, pos: 674
type: A, layer: 3, pos: 207
type: A, layer: 3, pos: 986
type: A, layer: 3, pos: 610
type: A, layer: 3, pos: 123
type: A, layer: 3, pos: 998
type: A, layer: 3, pos: 847
type: A, layer: 3, pos: 53
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 597
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 672
type: A, layer: 3, pos: 270
type: A, layer: 3, pos: 590
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 202
type: A, layer: 3, pos: 282
type: A, layer: 3, pos: 368
type: A, layer: 3, pos: 596
type: A, layer: 3, pos: 1001
type: A, layer: 3, pos: 321
type: A, layer: 3, pos: 364
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 834
type: A, layer: 3, pos: 126
type: A, layer: 3, pos: 970
type: A, layer: 3, pos: 362
type: A, layer: 3, pos: 275
type: A, layer: 3, pos: 973
type: A, layer: 3, pos: 1002
type: A, layer: 3, pos: 617
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 630
type: A, layer: 3, pos: 352
type: A, layer: 3, pos: 205
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 97
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 1006
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 276
type: A, layer: 3, pos: 606
type: A, layer: 3, pos: 52
type: A, layer: 3, pos: 112
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 845
type: A, layer: 3, pos: 602
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 588
type: A, layer: 3, pos: 119
type: A, layer: 3, pos: 320
type: A, layer: 3, pos: 215
type: A, layer: 3, pos: 217
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 593
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 871
type: A, layer: 3, pos: 979
type: A, layer: 3, pos: 594
type: A, layer: 3, pos: 629
type: A, layer: 3, pos: 301
type: A, layer: 3, pos: 1022
type: A, layer: 3, pos: 866
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 1008
type: A, layer: 3, pos: 89
type: A, layer: 3, pos: 114
type: A, layer: 3, pos: 87
type: A, layer: 3, pos: 862
type: A, layer: 3, pos: 587
type: A, layer: 3, pos: 580
type: A, layer: 3, pos: 88
type: A, layer: 3, pos: 966
type: A, layer: 3, pos: 210
type: A, layer: 3, pos: 852
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 960
type: A, layer: 3, pos: 694
type: A, layer: 3, pos: 692
type: A, layer: 3, pos: 371
type: A, layer: 3, pos: 1016
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 247
type: A, layer: 3, pos: 591
type: A, layer: 3, pos: 599
type: A, layer: 3, pos: 103
type: A, layer: 3, pos: 351
type: A, layer: 3, pos: 288
type: A, layer: 3, pos: 631
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 341
type: A, layer: 3, pos: 267
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 367
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 256
type: A, layer: 3, pos: 271
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 266
type: A, layer: 3, pos: 413
type: A, layer: 3, pos: 653
type: A, layer: 3, pos: 652
type: A, layer: 3, pos: 586
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 585
type: A, layer: 3, pos: 682
type: A, layer: 3, pos: 105
type: A, layer: 3, pos: 601
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 623
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 968
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 701
type: A, layer: 3, pos: 691
type: A, layer: 3, pos: 1011
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 685
type: A, layer: 3, pos: 125
type: A, layer: 3, pos: 639
type: A, layer: 3, pos: 650
type: A, layer: 3, pos: 74
type: A, layer: 3, pos: 687
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 680
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 269
type: A, layer: 3, pos: 285
type: A, layer: 3, pos: 303
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 618
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 609
type: A, layer: 3, pos: 622
type: A, layer: 3, pos: 636
type: A, layer: 3, pos: 405
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 607
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 243
type: A, layer: 3, pos: 626

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 236

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1787474, upper bound: 14.4825294
time: 58.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1787474, upper bound: 14.4825294
time: 56.59 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -37.4739952, -0.3004932, -37.5874672, -0.1140747, -37.3599205, 37.2869720
1: -17.5580101, 10.4648561, -17.6272316, 10.6045780, -28.1625881, 28.0920868
2: -14.2757969, 10.0701427, -14.4560261, 10.4401169, -24.7159138, 24.5261688
3: -14.7690849, 14.0477276, -14.9321232, 14.3925228, -29.1616077, 28.9798508
4: -14.9281874, 14.7359905, -15.1902523, 15.2530766, -30.1812630, 29.9262428
5: -14.0849934, 15.1739731, -14.2574596, 15.5613327, -29.6463261, 29.4314327
6: -20.7994633, 10.2554855, -20.9744606, 10.2845631, -31.0840263, 31.2299461
7: -17.2662201, 16.5131378, -17.4000130, 16.7180061, -33.6204720, 33.5483665
8: -16.1540527, 19.1408539, -16.3712387, 19.5482483, -35.6654930, 35.4744301
9: -15.1192532, 13.7080078, -15.2238607, 13.7518034, -28.6831474, 28.7379284
10: -23.4922657, 17.0078735, -23.9916286, 17.2991066, -40.7913742, 40.9995041
11: -26.1868706, 10.1109638, -26.7594604, 10.3938522, -36.5807228, 36.8704224
12: -24.2027283, 11.7994852, -24.9852009, 12.1842766, -36.3870049, 36.7846870
13: -22.1687183, 18.3957100, -22.2565498, 18.4694691, -40.6381874, 40.6522598
14: -47.8219643, -0.6076126, -48.2951813, -0.3876228, -47.2812119, 47.5353165
15: -19.5501804, 10.2969151, -19.6210079, 10.4508438, -30.0010242, 29.9179230
16: -24.9138050, 13.1436558, -25.1297760, 13.1591167, -37.6286278, 37.8146858
17: -43.9089127, 12.1483126, -44.5407104, 12.4716606, -55.0853882, 55.3927307
18: -20.4187355, 12.4415560, -20.4693527, 12.5114365, -32.9301720, 32.9109077
19: -17.8821640, 4.2019873, -18.1600037, 4.2943230, -22.1764870, 22.3619919
20: -15.2449818, 8.4316330, -15.3805351, 8.4879808, -23.7329636, 23.8121681
21: -25.8551636, 3.6564875, -26.2186642, 3.8075819, -29.6627464, 29.8751526
22: -32.9147339, -0.9469070, -33.0367813, -0.8464785, -30.7173882, 30.7283440
23: -17.8962727, 8.8600607, -18.1613483, 8.9529963, -26.8492699, 27.0214081
24: -25.2523575, 7.3256340, -25.2795830, 7.3797107, -31.1452484, 31.1138382
25: -18.2995110, 10.7651396, -18.4259834, 10.8650064, -29.1645164, 29.1911240
26: -23.6734314, 14.7520676, -23.9901695, 14.9273062, -38.6007385, 38.7422371
27: -26.2423801, 6.6745281, -26.2956467, 6.7200375, -31.9628220, 31.9713097
28: -17.2934952, 10.5998640, -17.4510975, 10.6396008, -27.7453461, 27.8609657
29: -40.1233482, -5.4482346, -40.4591026, -5.2593994, -33.9194946, 34.0755882
30: -20.8527565, 12.2628632, -21.0336304, 12.3281021, -33.1808586, 33.2964935
31: -23.6721535, 6.9580998, -23.8607845, 7.0183582, -30.6905117, 30.8188839
32: -27.6122303, 4.2924700, -27.8206062, 4.3980007, -31.1216393, 31.2324028
33: -30.4064388, 14.5593338, -30.5883980, 14.8846264, -44.3704758, 44.2252121
34: -25.9011497, 9.8880262, -26.0082226, 10.0228977, -35.9240494, 35.8962479
35: -27.6527424, 10.9415531, -27.7753048, 11.1205292, -38.3483658, 38.2864113
36: -27.1480255, 10.8774986, -27.2416286, 10.9005461, -37.6095390, 37.6727829
37: -37.1732521, 9.5992527, -37.3477478, 9.6360989, -45.5199432, 45.6639595
38: -29.6269455, 13.9693451, -29.7592144, 14.0492496, -43.6761932, 43.7285614
39: -38.3685722, 11.6025133, -38.5343170, 11.7582817, -49.4712677, 49.4855728
40: -30.3329468, 9.7854500, -30.4489994, 9.8675900, -38.5496826, 38.5809860
41: -22.3487549, 9.4839764, -22.5260963, 9.5027838, -31.8515396, 32.0100708
42: -16.3647785, 7.3994923, -16.7768135, 7.6182723, -23.6883507, 23.8873215

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 236
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 292
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 229
type: A, layer: 3, pos: 868
type: A, layer: 3, pos: 355
type: A, layer: 3, pos: 357
type: A, layer: 3, pos: 363
type: A, layer: 3, pos: 348
type: A, layer: 3, pos: 284
type: A, layer: 3, pos: 869
type: A, layer: 3, pos: 997
type: A, layer: 3, pos: 887
type: A, layer: 3, pos: 377
type: A, layer: 3, pos: 353
type: A, layer: 3, pos: 369
type: A, layer: 3, pos: 375
type: A, layer: 3, pos: 875
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 293
type: A, layer: 3, pos: 988
type: A, layer: 3, pos: 999
type: A, layer: 3, pos: 881
type: A, layer: 3, pos: 291
type: A, layer: 3, pos: 378
type: A, layer: 3, pos: 991
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 283
type: A, layer: 3, pos: 305
type: A, layer: 3, pos: 996
type: A, layer: 3, pos: 289
type: A, layer: 3, pos: 383
type: A, layer: 3, pos: 993
type: A, layer: 3, pos: 380
type: A, layer: 3, pos: 1009
type: A, layer: 3, pos: 893
type: A, layer: 3, pos: 877
type: A, layer: 3, pos: 331
type: A, layer: 3, pos: 361
type: A, layer: 3, pos: 339
type: A, layer: 3, pos: 849
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 338
type: A, layer: 3, pos: 311
type: A, layer: 3, pos: 1015
type: A, layer: 3, pos: 972
type: A, layer: 3, pos: 850
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 231
type: A, layer: 3, pos: 889
type: A, layer: 3, pos: 865
type: A, layer: 3, pos: 843
type: A, layer: 3, pos: 684
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 300
type: A, layer: 3, pos: 859
type: A, layer: 3, pos: 895
type: A, layer: 3, pos: 882
type: A, layer: 3, pos: 347
type: A, layer: 3, pos: 689
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 841
type: A, layer: 3, pos: 1023
type: A, layer: 3, pos: 379
type: A, layer: 3, pos: 382
type: A, layer: 3, pos: 644
type: A, layer: 3, pos: 860
type: A, layer: 3, pos: 695
type: A, layer: 3, pos: 223
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 346
type: A, layer: 3, pos: 329
type: A, layer: 3, pos: 306
type: A, layer: 3, pos: 265
type: A, layer: 3, pos: 724
type: A, layer: 3, pos: 1003
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 313
type: A, layer: 3, pos: 273
type: A, layer: 3, pos: 334
type: A, layer: 3, pos: 314
type: A, layer: 3, pos: 85
type: A, layer: 3, pos: 978
type: A, layer: 3, pos: 874
type: A, layer: 3, pos: 1005
type: A, layer: 3, pos: 58
type: A, layer: 3, pos: 1021
type: A, layer: 3, pos: 846
type: A, layer: 3, pos: 69
type: A, layer: 3, pos: 884
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 647
type: A, layer: 3, pos: 1017
type: A, layer: 3, pos: 699
type: A, layer: 3, pos: 274
type: A, layer: 3, pos: 977
type: A, layer: 3, pos: 299
type: A, layer: 3, pos: 894
type: A, layer: 3, pos: 974
type: A, layer: 3, pos: 995
type: A, layer: 3, pos: 370
type: A, layer: 3, pos: 851
type: A, layer: 3, pos: 646
type: A, layer: 3, pos: 698
type: A, layer: 3, pos: 876
type: A, layer: 3, pos: 667
type: A, layer: 3, pos: 260
type: A, layer: 3, pos: 381
type: A, layer: 3, pos: 235
type: A, layer: 3, pos: 673
type: A, layer: 3, pos: 1019
type: A, layer: 3, pos: 316
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 419
type: A, layer: 3, pos: 867
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 980
type: A, layer: 3, pos: 967
type: A, layer: 3, pos: 319
type: A, layer: 3, pos: 315
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 258
type: A, layer: 3, pos: 214
type: A, layer: 3, pos: 376
type: A, layer: 3, pos: 259
type: A, layer: 3, pos: 883
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 700
type: A, layer: 3, pos: 836
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 1020
type: A, layer: 3, pos: 842
type: A, layer: 3, pos: 61
type: A, layer: 3, pos: 1018
type: A, layer: 3, pos: 1014
type: A, layer: 3, pos: 336
type: A, layer: 3, pos: 688
type: A, layer: 3, pos: 56
type: A, layer: 3, pos: 272
type: A, layer: 3, pos: 1010
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 656
type: A, layer: 3, pos: 51
type: A, layer: 3, pos: 345
type: A, layer: 3, pos: 340
type: A, layer: 3, pos: 975
type: A, layer: 3, pos: 657
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 201
type: A, layer: 3, pos: 645
type: A, layer: 3, pos: 360
type: A, layer: 3, pos: 690
type: A, layer: 3, pos: 649
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 683
type: A, layer: 3, pos: 220
type: A, layer: 3, pos: 335
type: A, layer: 3, pos: 62
type: A, layer: 3, pos: 703
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 102
type: A, layer: 3, pos: 870
type: A, layer: 3, pos: 344
type: A, layer: 3, pos: 68
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 337
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 111
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 404
type: A, layer: 3, pos: 651
type: A, layer: 3, pos: 1013
type: A, layer: 3, pos: 349
type: A, layer: 3, pos: 858
type: A, layer: 3, pos: 981
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 242
type: A, layer: 3, pos: 861
type: A, layer: 3, pos: 1004
type: A, layer: 3, pos: 987
type: A, layer: 3, pos: 279
type: A, layer: 3, pos: 325
type: A, layer: 3, pos: 113
type: A, layer: 3, pos: 281
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 78
type: A, layer: 3, pos: 658
type: A, layer: 3, pos: 57
type: A, layer: 3, pos: 54
type: A, layer: 3, pos: 297
type: A, layer: 3, pos: 203
type: A, layer: 3, pos: 835
type: A, layer: 3, pos: 971
type: A, layer: 3, pos: 420
type: A, layer: 3, pos: 63
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 55
type: A, layer: 3, pos: 879
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 595
type: A, layer: 3, pos: 969
type: A, layer: 3, pos: 263
type: A, layer: 3, pos: 642
type: A, layer: 3, pos: 702
type: A, layer: 3, pos: 318
type: A, layer: 3, pos: 863
type: A, layer: 3, pos: 983
type: A, layer: 3, pos: 328
type: A, layer: 3, pos: 257
type: A, layer: 3, pos: 675
type: A, layer: 3, pos: 50
type: A, layer: 3, pos: 343
type: A, layer: 3, pos: 965
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 365
type: A, layer: 3, pos: 855
type: A, layer: 3, pos: 428
type: A, layer: 3, pos: 664
type: A, layer: 3, pos: 86
type: A, layer: 3, pos: 246
type: A, layer: 3, pos: 354
type: A, layer: 3, pos: 252
type: A, layer: 3, pos: 598
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 643
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 238
type: A, layer: 3, pos: 333
type: A, layer: 3, pos: 1012
type: A, layer: 3, pos: 264
type: A, layer: 3, pos: 982
type: A, layer: 3, pos: 124
type: A, layer: 3, pos: 262
type: A, layer: 3, pos: 648
type: A, layer: 3, pos: 641
type: A, layer: 3, pos: 985
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 77
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 857
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 372
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 666
type: A, layer: 3, pos: 322
type: A, layer: 3, pos: 84
type: A, layer: 3, pos: 873
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 665
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 324
type: A, layer: 3, pos: 109
type: A, layer: 3, pos: 844
type: A, layer: 3, pos: 1007
type: A, layer: 3, pos: 82
type: A, layer: 3, pos: 589
type: A, layer: 3, pos: 663
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 696
type: A, layer: 3, pos: 209
type: A, layer: 3, pos: 94
type: A, layer: 3, pos: 296
type: A, layer: 3, pos: 989
type: A, layer: 3, pos: 885
type: A, layer: 3, pos: 251
type: A, layer: 3, pos: 990
type: A, layer: 3, pos: 104
type: A, layer: 3, pos: 681
type: A, layer: 3, pos: 261
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 358
type: A, layer: 3, pos: 127
type: A, layer: 3, pos: 304
type: A, layer: 3, pos: 280
type: A, layer: 3, pos: 628
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 853
type: A, layer: 3, pos: 249
type: A, layer: 3, pos: 833
type: A, layer: 3, pos: 674
type: A, layer: 3, pos: 207
type: A, layer: 3, pos: 986
type: A, layer: 3, pos: 610
type: A, layer: 3, pos: 123
type: A, layer: 3, pos: 998
type: A, layer: 3, pos: 847
type: A, layer: 3, pos: 53
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 597
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 672
type: A, layer: 3, pos: 270
type: A, layer: 3, pos: 590
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 202
type: A, layer: 3, pos: 282
type: A, layer: 3, pos: 368
type: A, layer: 3, pos: 596
type: A, layer: 3, pos: 1001
type: A, layer: 3, pos: 321
type: A, layer: 3, pos: 364
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 834
type: A, layer: 3, pos: 126
type: A, layer: 3, pos: 970
type: A, layer: 3, pos: 362
type: A, layer: 3, pos: 275
type: A, layer: 3, pos: 973
type: A, layer: 3, pos: 1002
type: A, layer: 3, pos: 617
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 630
type: A, layer: 3, pos: 352
type: A, layer: 3, pos: 205
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 97
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 1006
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 276
type: A, layer: 3, pos: 606
type: A, layer: 3, pos: 52
type: A, layer: 3, pos: 112
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 845
type: A, layer: 3, pos: 602
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 588
type: A, layer: 3, pos: 119
type: A, layer: 3, pos: 320
type: A, layer: 3, pos: 215
type: A, layer: 3, pos: 217
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 593
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 871
type: A, layer: 3, pos: 979
type: A, layer: 3, pos: 594
type: A, layer: 3, pos: 629
type: A, layer: 3, pos: 301
type: A, layer: 3, pos: 1022
type: A, layer: 3, pos: 866
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 1008
type: A, layer: 3, pos: 89
type: A, layer: 3, pos: 114
type: A, layer: 3, pos: 87
type: A, layer: 3, pos: 862
type: A, layer: 3, pos: 587
type: A, layer: 3, pos: 580
type: A, layer: 3, pos: 88
type: A, layer: 3, pos: 966
type: A, layer: 3, pos: 210
type: A, layer: 3, pos: 852
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 960
type: A, layer: 3, pos: 694
type: A, layer: 3, pos: 692
type: A, layer: 3, pos: 371
type: A, layer: 3, pos: 1016
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 247
type: A, layer: 3, pos: 591
type: A, layer: 3, pos: 599
type: A, layer: 3, pos: 103
type: A, layer: 3, pos: 351
type: A, layer: 3, pos: 288
type: A, layer: 3, pos: 631
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 341
type: A, layer: 3, pos: 267
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 367
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 256
type: A, layer: 3, pos: 271
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 266
type: A, layer: 3, pos: 413
type: A, layer: 3, pos: 653
type: A, layer: 3, pos: 652
type: A, layer: 3, pos: 586
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 585
type: A, layer: 3, pos: 682
type: A, layer: 3, pos: 105
type: A, layer: 3, pos: 601
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 623
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 968
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 701
type: A, layer: 3, pos: 691
type: A, layer: 3, pos: 1011
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 685
type: A, layer: 3, pos: 125
type: A, layer: 3, pos: 639
type: A, layer: 3, pos: 650
type: A, layer: 3, pos: 74
type: A, layer: 3, pos: 687
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 680
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 269
type: A, layer: 3, pos: 285
type: A, layer: 3, pos: 303
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 618
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 609
type: A, layer: 3, pos: 622
type: A, layer: 3, pos: 636
type: A, layer: 3, pos: 405
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 607
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 243
type: A, layer: 3, pos: 626

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 236

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1787474, upper bound: 14.6499224
time: 59.85 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1787474, upper bound: 14.6499224
time: 57.91 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -37.5717468, -0.1190395, -37.5071754, -0.3091259, -37.2626190, 37.3881378
1: -17.6085167, 10.5991268, -17.5642986, 10.4647131, -28.0732307, 28.1634254
2: -14.4205818, 10.4355049, -14.3237028, 10.0066261, -24.4272079, 24.7592087
3: -14.9194307, 14.3844986, -14.8717422, 14.0338211, -28.9532509, 29.2562408
4: -15.1261616, 15.2482014, -14.9305439, 14.6181946, -29.7443562, 30.1787453
5: -14.2462921, 15.5564632, -14.1866379, 15.1604195, -29.4067116, 29.7431011
6: -20.9641838, 10.2232609, -20.6757736, 10.1171150, -31.0812988, 30.8990345
7: -17.3878899, 16.7149620, -17.3189659, 16.5156002, -33.5257568, 33.6710205
8: -16.3111744, 19.5392342, -16.1338844, 19.0292397, -35.2755928, 35.6406441
9: -15.2108641, 13.7007027, -15.0199347, 13.5914555, -28.6109352, 28.5456123
10: -23.9720459, 17.1925964, -23.2388115, 16.9372902, -40.9093361, 40.4314079
11: -26.7462006, 10.3291969, -26.0784836, 10.1660213, -36.9122238, 36.4076805
12: -24.9791145, 12.1248417, -24.0856400, 11.9288559, -36.9079704, 36.2104797
13: -22.2509842, 18.4505997, -22.1650906, 18.3709202, -40.6219025, 40.6156921
14: -48.2812462, -0.3966770, -47.8159828, -0.4266815, -47.6997070, 47.2471046
15: -19.5713100, 10.4389124, -19.4617043, 10.2652588, -29.8365688, 29.9006157
16: -25.1079731, 13.0775785, -24.7754993, 12.9723110, -37.6334305, 37.4157562
17: -44.5339813, 12.4492607, -43.9194069, 12.3462505, -55.5840836, 55.0677872
18: -20.4511070, 12.4966764, -20.3621769, 12.4415760, -32.8926849, 32.8588524
19: -18.1492996, 4.2843523, -17.8593864, 4.2554235, -22.4047241, 22.1437378
20: -15.3727322, 8.4819546, -15.2400579, 8.4609413, -23.8336735, 23.7220116
21: -26.2082367, 3.7971272, -25.8254356, 3.7555432, -29.9637794, 29.6225624
22: -32.9963989, -0.8563404, -32.8242607, -0.9364538, -30.8107300, 30.6169548
23: -18.1549244, 8.9355650, -17.8813972, 8.8835506, -27.0384750, 26.8169632
24: -25.2368450, 7.3771935, -25.1438484, 7.2240210, -31.0014725, 31.0311985
25: -18.4059219, 10.8596420, -18.2527542, 10.8019829, -29.2079048, 29.1123962
26: -23.9739456, 14.9131374, -23.6457214, 14.8653002, -38.8392448, 38.5588608
27: -26.2566891, 6.7151709, -26.1476631, 6.6257153, -31.9188766, 31.8593235
28: -17.4452858, 10.6351357, -17.2946835, 10.6096840, -27.8642654, 27.7307262
29: -40.4464073, -5.2655640, -40.1154785, -5.3037758, -34.2302856, 33.8835487
30: -21.0250111, 12.3176250, -20.8438377, 12.2933455, -33.3183556, 33.1614609
31: -23.8461189, 7.0141349, -23.6397247, 6.9866867, -30.8328056, 30.6538601
32: -27.8142204, 4.3573856, -27.5307484, 4.2602167, -31.1963196, 30.9750938
33: -30.5583839, 14.8763590, -30.4479084, 14.5324764, -44.1325760, 44.4720840
34: -25.9964447, 10.0103292, -25.9322491, 9.8801212, -35.8765640, 35.9425774
35: -27.7376995, 11.1107368, -27.6307526, 10.8708572, -38.1560211, 38.4174194
36: -27.2275429, 10.8934174, -27.1271973, 10.8685951, -37.6584282, 37.5790482
37: -37.3332291, 9.6304016, -37.1669502, 9.6051941, -45.6799927, 45.5065575
38: -29.7362843, 14.0384769, -29.6233559, 13.9542847, -43.6905670, 43.6618347
39: -38.4966736, 11.7549076, -38.3230209, 11.5742073, -49.4229279, 49.4211884
40: -30.4348030, 9.8576508, -30.3736992, 9.7646046, -38.5429955, 38.5717888
41: -22.5147266, 9.4669914, -22.2877579, 9.4177570, -31.9324837, 31.7547493
42: -16.7699909, 7.5357118, -16.1838455, 7.3236570, -23.8095284, 23.3886528

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 292
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 229
type: A, layer: 3, pos: 868
type: A, layer: 3, pos: 355
type: A, layer: 3, pos: 236
type: A, layer: 3, pos: 357
type: A, layer: 3, pos: 363
type: A, layer: 3, pos: 348
type: A, layer: 3, pos: 284
type: A, layer: 3, pos: 869
type: A, layer: 3, pos: 997
type: A, layer: 3, pos: 887
type: A, layer: 3, pos: 377
type: A, layer: 3, pos: 353
type: A, layer: 3, pos: 369
type: A, layer: 3, pos: 375
type: A, layer: 3, pos: 875
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 293
type: A, layer: 3, pos: 988
type: A, layer: 3, pos: 999
type: A, layer: 3, pos: 881
type: A, layer: 3, pos: 291
type: A, layer: 3, pos: 378
type: A, layer: 3, pos: 991
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 283
type: A, layer: 3, pos: 305
type: A, layer: 3, pos: 996
type: A, layer: 3, pos: 289
type: A, layer: 3, pos: 383
type: A, layer: 3, pos: 993
type: A, layer: 3, pos: 380
type: A, layer: 3, pos: 1009
type: A, layer: 3, pos: 893
type: A, layer: 3, pos: 331
type: A, layer: 3, pos: 877
type: A, layer: 3, pos: 361
type: A, layer: 3, pos: 339
type: A, layer: 3, pos: 849
type: A, layer: 3, pos: 338
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 311
type: A, layer: 3, pos: 1015
type: A, layer: 3, pos: 972
type: A, layer: 3, pos: 850
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 231
type: A, layer: 3, pos: 889
type: A, layer: 3, pos: 865
type: A, layer: 3, pos: 843
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 684
type: A, layer: 3, pos: 300
type: A, layer: 3, pos: 859
type: A, layer: 3, pos: 895
type: A, layer: 3, pos: 882
type: A, layer: 3, pos: 347
type: A, layer: 3, pos: 689
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 841
type: A, layer: 3, pos: 1023
type: A, layer: 3, pos: 379
type: A, layer: 3, pos: 382
type: A, layer: 3, pos: 644
type: A, layer: 3, pos: 860
type: A, layer: 3, pos: 695
type: A, layer: 3, pos: 223
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 346
type: A, layer: 3, pos: 329
type: A, layer: 3, pos: 265
type: A, layer: 3, pos: 306
type: A, layer: 3, pos: 1003
type: A, layer: 3, pos: 724
type: A, layer: 3, pos: 313
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 273
type: A, layer: 3, pos: 314
type: A, layer: 3, pos: 334
type: A, layer: 3, pos: 85
type: A, layer: 3, pos: 978
type: A, layer: 3, pos: 1005
type: A, layer: 3, pos: 874
type: A, layer: 3, pos: 58
type: A, layer: 3, pos: 1021
type: A, layer: 3, pos: 846
type: A, layer: 3, pos: 69
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 884
type: A, layer: 3, pos: 647
type: A, layer: 3, pos: 1017
type: A, layer: 3, pos: 699
type: A, layer: 3, pos: 274
type: A, layer: 3, pos: 977
type: A, layer: 3, pos: 299
type: A, layer: 3, pos: 894
type: A, layer: 3, pos: 995
type: A, layer: 3, pos: 974
type: A, layer: 3, pos: 851
type: A, layer: 3, pos: 370
type: A, layer: 3, pos: 646
type: A, layer: 3, pos: 876
type: A, layer: 3, pos: 698
type: A, layer: 3, pos: 667
type: A, layer: 3, pos: 260
type: A, layer: 3, pos: 381
type: A, layer: 3, pos: 673
type: A, layer: 3, pos: 235
type: A, layer: 3, pos: 1019
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 316
type: A, layer: 3, pos: 419
type: A, layer: 3, pos: 867
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 980
type: A, layer: 3, pos: 319
type: A, layer: 3, pos: 967
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 258
type: A, layer: 3, pos: 214
type: A, layer: 3, pos: 376
type: A, layer: 3, pos: 315
type: A, layer: 3, pos: 259
type: A, layer: 3, pos: 883
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 700
type: A, layer: 3, pos: 836
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 1020
type: A, layer: 3, pos: 842
type: A, layer: 3, pos: 61
type: A, layer: 3, pos: 1018
type: A, layer: 3, pos: 336
type: A, layer: 3, pos: 688
type: A, layer: 3, pos: 1014
type: A, layer: 3, pos: 56
type: A, layer: 3, pos: 272
type: A, layer: 3, pos: 1010
type: A, layer: 3, pos: 656
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 345
type: A, layer: 3, pos: 51
type: A, layer: 3, pos: 340
type: A, layer: 3, pos: 975
type: A, layer: 3, pos: 657
type: A, layer: 3, pos: 201
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 645
type: A, layer: 3, pos: 649
type: A, layer: 3, pos: 360
type: A, layer: 3, pos: 690
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 683
type: A, layer: 3, pos: 335
type: A, layer: 3, pos: 220
type: A, layer: 3, pos: 703
type: A, layer: 3, pos: 62
type: A, layer: 3, pos: 102
type: A, layer: 3, pos: 870
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 344
type: A, layer: 3, pos: 68
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 337
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 111
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 404
type: A, layer: 3, pos: 1013
type: A, layer: 3, pos: 651
type: A, layer: 3, pos: 981
type: A, layer: 3, pos: 858
type: A, layer: 3, pos: 349
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 861
type: A, layer: 3, pos: 242
type: A, layer: 3, pos: 1004
type: A, layer: 3, pos: 325
type: A, layer: 3, pos: 279
type: A, layer: 3, pos: 987
type: A, layer: 3, pos: 113
type: A, layer: 3, pos: 281
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 78
type: A, layer: 3, pos: 658
type: A, layer: 3, pos: 54
type: A, layer: 3, pos: 57
type: A, layer: 3, pos: 297
type: A, layer: 3, pos: 203
type: A, layer: 3, pos: 835
type: A, layer: 3, pos: 971
type: A, layer: 3, pos: 420
type: A, layer: 3, pos: 63
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 879
type: A, layer: 3, pos: 55
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 969
type: A, layer: 3, pos: 263
type: A, layer: 3, pos: 318
type: A, layer: 3, pos: 595
type: A, layer: 3, pos: 642
type: A, layer: 3, pos: 702
type: A, layer: 3, pos: 863
type: A, layer: 3, pos: 983
type: A, layer: 3, pos: 257
type: A, layer: 3, pos: 328
type: A, layer: 3, pos: 675
type: A, layer: 3, pos: 50
type: A, layer: 3, pos: 343
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 365
type: A, layer: 3, pos: 965
type: A, layer: 3, pos: 855
type: A, layer: 3, pos: 664
type: A, layer: 3, pos: 252
type: A, layer: 3, pos: 354
type: A, layer: 3, pos: 428
type: A, layer: 3, pos: 86
type: A, layer: 3, pos: 246
type: A, layer: 3, pos: 598
type: A, layer: 3, pos: 643
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 238
type: A, layer: 3, pos: 333
type: A, layer: 3, pos: 262
type: A, layer: 3, pos: 982
type: A, layer: 3, pos: 264
type: A, layer: 3, pos: 1012
type: A, layer: 3, pos: 124
type: A, layer: 3, pos: 648
type: A, layer: 3, pos: 641
type: A, layer: 3, pos: 985
type: A, layer: 3, pos: 857
type: A, layer: 3, pos: 77
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 372
type: A, layer: 3, pos: 666
type: A, layer: 3, pos: 322
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 84
type: A, layer: 3, pos: 873
type: A, layer: 3, pos: 665
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 324
type: A, layer: 3, pos: 109
type: A, layer: 3, pos: 1007
type: A, layer: 3, pos: 844
type: A, layer: 3, pos: 82
type: A, layer: 3, pos: 663
type: A, layer: 3, pos: 589
type: A, layer: 3, pos: 696
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 209
type: A, layer: 3, pos: 94
type: A, layer: 3, pos: 296
type: A, layer: 3, pos: 990
type: A, layer: 3, pos: 251
type: A, layer: 3, pos: 885
type: A, layer: 3, pos: 989
type: A, layer: 3, pos: 261
type: A, layer: 3, pos: 104
type: A, layer: 3, pos: 681
type: A, layer: 3, pos: 853
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 358
type: A, layer: 3, pos: 304
type: A, layer: 3, pos: 280
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 127
type: A, layer: 3, pos: 628
type: A, layer: 3, pos: 249
type: A, layer: 3, pos: 674
type: A, layer: 3, pos: 833
type: A, layer: 3, pos: 998
type: A, layer: 3, pos: 986
type: A, layer: 3, pos: 610
type: A, layer: 3, pos: 207
type: A, layer: 3, pos: 123
type: A, layer: 3, pos: 847
type: A, layer: 3, pos: 53
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 597
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 672
type: A, layer: 3, pos: 270
type: A, layer: 3, pos: 590
type: A, layer: 3, pos: 202
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 282
type: A, layer: 3, pos: 368
type: A, layer: 3, pos: 596
type: A, layer: 3, pos: 1001
type: A, layer: 3, pos: 364
type: A, layer: 3, pos: 321
type: A, layer: 3, pos: 126
type: A, layer: 3, pos: 834
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 362
type: A, layer: 3, pos: 970
type: A, layer: 3, pos: 275
type: A, layer: 3, pos: 1002
type: A, layer: 3, pos: 617
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 973
type: A, layer: 3, pos: 630
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 352
type: A, layer: 3, pos: 205
type: A, layer: 3, pos: 97
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 1006
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 606
type: A, layer: 3, pos: 52
type: A, layer: 3, pos: 276
type: A, layer: 3, pos: 112
type: A, layer: 3, pos: 845
type: A, layer: 3, pos: 602
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 119
type: A, layer: 3, pos: 215
type: A, layer: 3, pos: 588
type: A, layer: 3, pos: 320
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 217
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 593
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 871
type: A, layer: 3, pos: 301
type: A, layer: 3, pos: 979
type: A, layer: 3, pos: 594
type: A, layer: 3, pos: 629
type: A, layer: 3, pos: 1022
type: A, layer: 3, pos: 866
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 1008
type: A, layer: 3, pos: 89
type: A, layer: 3, pos: 114
type: A, layer: 3, pos: 87
type: A, layer: 3, pos: 587
type: A, layer: 3, pos: 862
type: A, layer: 3, pos: 580
type: A, layer: 3, pos: 88
type: A, layer: 3, pos: 966
type: A, layer: 3, pos: 210
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 960
type: A, layer: 3, pos: 852
type: A, layer: 3, pos: 694
type: A, layer: 3, pos: 692
type: A, layer: 3, pos: 371
type: A, layer: 3, pos: 1016
type: A, layer: 3, pos: 247
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 591
type: A, layer: 3, pos: 599
type: A, layer: 3, pos: 103
type: A, layer: 3, pos: 288
type: A, layer: 3, pos: 351
type: A, layer: 3, pos: 341
type: A, layer: 3, pos: 631
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 267
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 367
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 256
type: A, layer: 3, pos: 266
type: A, layer: 3, pos: 271
type: A, layer: 3, pos: 413
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 653
type: A, layer: 3, pos: 652
type: A, layer: 3, pos: 586
type: A, layer: 3, pos: 585
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 682
type: A, layer: 3, pos: 105
type: A, layer: 3, pos: 601
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 623
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 968
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 701
type: A, layer: 3, pos: 691
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 1011
type: A, layer: 3, pos: 685
type: A, layer: 3, pos: 125
type: A, layer: 3, pos: 639
type: A, layer: 3, pos: 74
type: A, layer: 3, pos: 650
type: A, layer: 3, pos: 687
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 680
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 269
type: A, layer: 3, pos: 285
type: A, layer: 3, pos: 303
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 609
type: A, layer: 3, pos: 618
type: A, layer: 3, pos: 622
type: A, layer: 3, pos: 636
type: A, layer: 3, pos: 405
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 243
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 607
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 626

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 237

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1389150, upper bound: 14.5140414
time: 58.52 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1389150, upper bound: 14.5140414
time: 65.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -37.5809860, -0.1156368, -37.5336418, -0.2879257, -37.2930603, 37.4180069
1: -17.6196594, 10.6034050, -17.5910110, 10.4767389, -28.0963974, 28.1944160
2: -14.4486475, 10.4392662, -14.4068851, 10.0843773, -24.5330238, 24.8461514
3: -14.9279366, 14.3909569, -14.8979521, 14.0701256, -28.9980621, 29.2889099
4: -15.1784153, 15.2519703, -15.0952349, 14.7505436, -29.9289589, 30.3472061
5: -14.2539558, 15.5601444, -14.2123756, 15.1893301, -29.4432869, 29.7725201
6: -20.9720974, 10.2793941, -20.8247643, 10.2788963, -31.2509937, 31.1041584
7: -17.3954506, 16.7170830, -17.3482475, 16.5251122, -33.5442963, 33.7021866
8: -16.3608932, 19.5460815, -16.2875671, 19.1622925, -35.4999695, 35.8028908
9: -15.2220650, 13.7468815, -15.1553631, 13.7330027, -28.7251816, 28.7159882
10: -23.9891758, 17.2923393, -23.5382347, 17.2245674, -41.2137451, 40.8305740
11: -26.7565308, 10.3782673, -26.2165775, 10.3115959, -37.0681267, 36.5948448
12: -24.9833317, 12.1751394, -24.2276077, 12.0799484, -37.0632782, 36.4027481
13: -22.2552299, 18.4635239, -22.1918640, 18.4182816, -40.6735115, 40.6553879
14: -48.2837677, -0.3895340, -47.8323021, -0.4300365, -47.6937256, 47.2722855
15: -19.6061230, 10.4485245, -19.5678024, 10.3200626, -29.9261856, 30.0163269
16: -25.1271172, 13.1470222, -24.9667168, 13.1723309, -37.8090363, 37.6683006
17: -44.5383377, 12.4662628, -43.9523163, 12.4030018, -55.6451569, 55.1174164
18: -20.4603729, 12.5098648, -20.4027138, 12.4793091, -32.9396820, 32.9125786
19: -18.1575508, 4.2917461, -17.9089470, 4.2763166, -22.4338684, 22.2006931
20: -15.3783112, 8.4861803, -15.2681856, 8.4709473, -23.8492584, 23.7543659
21: -26.2160244, 3.8045201, -25.8814735, 3.7749612, -29.9909859, 29.6859932
22: -33.0216637, -0.8478050, -32.9059868, -0.8845272, -30.8398705, 30.6917400
23: -18.1596069, 8.9482460, -17.9175262, 8.9236813, -27.0832882, 26.8657722
24: -25.2722054, 7.3790770, -25.2493420, 7.3397903, -31.1386452, 31.0562420
25: -18.4217739, 10.8634243, -18.3104115, 10.8200722, -29.2418461, 29.1738358
26: -23.9773865, 14.9257889, -23.6703796, 14.8914185, -38.8688049, 38.5961685
27: -26.2841721, 6.7192240, -26.2387104, 6.6846504, -32.0110855, 31.9215050
28: -17.4492397, 10.6384287, -17.3122177, 10.6288843, -27.8838425, 27.7566414
29: -40.4520340, -5.2605515, -40.1408501, -5.2895088, -34.2342453, 33.9758949
30: -21.0314293, 12.3227940, -20.8691139, 12.3123703, -33.3437996, 33.1919098
31: -23.8573780, 7.0166903, -23.6997452, 6.9949493, -30.8523273, 30.7164345
32: -27.8191795, 4.3942890, -27.6320686, 4.3685112, -31.2791672, 31.1221886
33: -30.5784225, 14.8826962, -30.5085564, 14.5920391, -44.2309189, 44.5350800
34: -26.0028572, 10.0214005, -25.9544525, 9.9210644, -35.9239197, 35.9758530
35: -27.7671261, 11.1190071, -27.7151680, 10.9629478, -38.2742653, 38.4973450
36: -27.2356243, 10.8993835, -27.1565666, 10.8941479, -37.6902771, 37.6118660
37: -37.3436279, 9.6348019, -37.2069626, 9.6254301, -45.7079315, 45.5481987
38: -29.7505569, 14.0479794, -29.6707153, 13.9914112, -43.7419662, 43.7186966
39: -38.5248260, 11.7573090, -38.4189377, 11.6169243, -49.4900589, 49.5059471
40: -30.4461555, 9.8499193, -30.3898697, 9.7419548, -38.5400276, 38.5820694
41: -22.5235271, 9.4964552, -22.3843689, 9.4993410, -32.0052948, 31.8808250
42: -16.7751427, 7.6122546, -16.3891487, 7.5523911, -23.9127388, 23.6694145

Time for backsubstitution: 0.98 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 292
type: A, layer: 3, pos: 228
type: A, layer: 3, pos: 229
type: A, layer: 3, pos: 868
type: A, layer: 3, pos: 355
type: A, layer: 3, pos: 236
type: A, layer: 3, pos: 357
type: A, layer: 3, pos: 363
type: A, layer: 3, pos: 348
type: A, layer: 3, pos: 284
type: A, layer: 3, pos: 869
type: A, layer: 3, pos: 997
type: A, layer: 3, pos: 887
type: A, layer: 3, pos: 377
type: A, layer: 3, pos: 353
type: A, layer: 3, pos: 369
type: A, layer: 3, pos: 375
type: A, layer: 3, pos: 875
type: A, layer: 3, pos: 892
type: A, layer: 3, pos: 293
type: A, layer: 3, pos: 988
type: A, layer: 3, pos: 999
type: A, layer: 3, pos: 881
type: A, layer: 3, pos: 291
type: A, layer: 3, pos: 378
type: A, layer: 3, pos: 991
type: A, layer: 3, pos: 668
type: A, layer: 3, pos: 305
type: A, layer: 3, pos: 283
type: A, layer: 3, pos: 996
type: A, layer: 3, pos: 289
type: A, layer: 3, pos: 383
type: A, layer: 3, pos: 993
type: A, layer: 3, pos: 380
type: A, layer: 3, pos: 1009
type: A, layer: 3, pos: 893
type: A, layer: 3, pos: 331
type: A, layer: 3, pos: 877
type: A, layer: 3, pos: 361
type: A, layer: 3, pos: 339
type: A, layer: 3, pos: 849
type: A, layer: 3, pos: 338
type: A, layer: 3, pos: 891
type: A, layer: 3, pos: 311
type: A, layer: 3, pos: 1015
type: A, layer: 3, pos: 850
type: A, layer: 3, pos: 972
type: A, layer: 3, pos: 676
type: A, layer: 3, pos: 231
type: A, layer: 3, pos: 889
type: A, layer: 3, pos: 865
type: A, layer: 3, pos: 843
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 684
type: A, layer: 3, pos: 300
type: A, layer: 3, pos: 859
type: A, layer: 3, pos: 895
type: A, layer: 3, pos: 882
type: A, layer: 3, pos: 347
type: A, layer: 3, pos: 689
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 841
type: A, layer: 3, pos: 1023
type: A, layer: 3, pos: 379
type: A, layer: 3, pos: 382
type: A, layer: 3, pos: 644
type: A, layer: 3, pos: 860
type: A, layer: 3, pos: 695
type: A, layer: 3, pos: 223
type: A, layer: 3, pos: 239
type: A, layer: 3, pos: 346
type: A, layer: 3, pos: 329
type: A, layer: 3, pos: 265
type: A, layer: 3, pos: 306
type: A, layer: 3, pos: 1003
type: A, layer: 3, pos: 724
type: A, layer: 3, pos: 313
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 273
type: A, layer: 3, pos: 314
type: A, layer: 3, pos: 334
type: A, layer: 3, pos: 85
type: A, layer: 3, pos: 978
type: A, layer: 3, pos: 1005
type: A, layer: 3, pos: 874
type: A, layer: 3, pos: 58
type: A, layer: 3, pos: 1021
type: A, layer: 3, pos: 846
type: A, layer: 3, pos: 69
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 884
type: A, layer: 3, pos: 647
type: A, layer: 3, pos: 1017
type: A, layer: 3, pos: 699
type: A, layer: 3, pos: 274
type: A, layer: 3, pos: 977
type: A, layer: 3, pos: 299
type: A, layer: 3, pos: 894
type: A, layer: 3, pos: 995
type: A, layer: 3, pos: 974
type: A, layer: 3, pos: 370
type: A, layer: 3, pos: 851
type: A, layer: 3, pos: 646
type: A, layer: 3, pos: 698
type: A, layer: 3, pos: 876
type: A, layer: 3, pos: 667
type: A, layer: 3, pos: 260
type: A, layer: 3, pos: 381
type: A, layer: 3, pos: 673
type: A, layer: 3, pos: 235
type: A, layer: 3, pos: 1019
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 316
type: A, layer: 3, pos: 419
type: A, layer: 3, pos: 867
type: A, layer: 3, pos: 204
type: A, layer: 3, pos: 980
type: A, layer: 3, pos: 319
type: A, layer: 3, pos: 967
type: A, layer: 3, pos: 697
type: A, layer: 3, pos: 258
type: A, layer: 3, pos: 214
type: A, layer: 3, pos: 376
type: A, layer: 3, pos: 315
type: A, layer: 3, pos: 259
type: A, layer: 3, pos: 883
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 700
type: A, layer: 3, pos: 836
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 1020
type: A, layer: 3, pos: 842
type: A, layer: 3, pos: 61
type: A, layer: 3, pos: 1018
type: A, layer: 3, pos: 336
type: A, layer: 3, pos: 688
type: A, layer: 3, pos: 1014
type: A, layer: 3, pos: 56
type: A, layer: 3, pos: 272
type: A, layer: 3, pos: 1010
type: A, layer: 3, pos: 656
type: A, layer: 3, pos: 255
type: A, layer: 3, pos: 345
type: A, layer: 3, pos: 51
type: A, layer: 3, pos: 340
type: A, layer: 3, pos: 975
type: A, layer: 3, pos: 657
type: A, layer: 3, pos: 201
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 645
type: A, layer: 3, pos: 649
type: A, layer: 3, pos: 360
type: A, layer: 3, pos: 690
type: A, layer: 3, pos: 604
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 683
type: A, layer: 3, pos: 335
type: A, layer: 3, pos: 220
type: A, layer: 3, pos: 703
type: A, layer: 3, pos: 62
type: A, layer: 3, pos: 870
type: A, layer: 3, pos: 102
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 344
type: A, layer: 3, pos: 68
type: A, layer: 3, pos: 71
type: A, layer: 3, pos: 337
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 111
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 59
type: A, layer: 3, pos: 404
type: A, layer: 3, pos: 651
type: A, layer: 3, pos: 1013
type: A, layer: 3, pos: 981
type: A, layer: 3, pos: 349
type: A, layer: 3, pos: 858
type: A, layer: 3, pos: 411
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 861
type: A, layer: 3, pos: 242
type: A, layer: 3, pos: 1004
type: A, layer: 3, pos: 279
type: A, layer: 3, pos: 325
type: A, layer: 3, pos: 987
type: A, layer: 3, pos: 113
type: A, layer: 3, pos: 281
type: A, layer: 3, pos: 317
type: A, layer: 3, pos: 95
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 233
type: A, layer: 3, pos: 225
type: A, layer: 3, pos: 78
type: A, layer: 3, pos: 658
type: A, layer: 3, pos: 54
type: A, layer: 3, pos: 57
type: A, layer: 3, pos: 297
type: A, layer: 3, pos: 203
type: A, layer: 3, pos: 835
type: A, layer: 3, pos: 971
type: A, layer: 3, pos: 420
type: A, layer: 3, pos: 63
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 879
type: A, layer: 3, pos: 55
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 969
type: A, layer: 3, pos: 263
type: A, layer: 3, pos: 318
type: A, layer: 3, pos: 595
type: A, layer: 3, pos: 642
type: A, layer: 3, pos: 702
type: A, layer: 3, pos: 863
type: A, layer: 3, pos: 983
type: A, layer: 3, pos: 257
type: A, layer: 3, pos: 328
type: A, layer: 3, pos: 675
type: A, layer: 3, pos: 50
type: A, layer: 3, pos: 343
type: A, layer: 3, pos: 206
type: A, layer: 3, pos: 965
type: A, layer: 3, pos: 365
type: A, layer: 3, pos: 354
type: A, layer: 3, pos: 664
type: A, layer: 3, pos: 855
type: A, layer: 3, pos: 252
type: A, layer: 3, pos: 428
type: A, layer: 3, pos: 86
type: A, layer: 3, pos: 246
type: A, layer: 3, pos: 598
type: A, layer: 3, pos: 677
type: A, layer: 3, pos: 643
type: A, layer: 3, pos: 238
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 333
type: A, layer: 3, pos: 262
type: A, layer: 3, pos: 264
type: A, layer: 3, pos: 982
type: A, layer: 3, pos: 1012
type: A, layer: 3, pos: 124
type: A, layer: 3, pos: 648
type: A, layer: 3, pos: 641
type: A, layer: 3, pos: 857
type: A, layer: 3, pos: 985
type: A, layer: 3, pos: 72
type: A, layer: 3, pos: 77
type: A, layer: 3, pos: 79
type: A, layer: 3, pos: 122
type: A, layer: 3, pos: 29
type: A, layer: 3, pos: 372
type: A, layer: 3, pos: 666
type: A, layer: 3, pos: 322
type: A, layer: 3, pos: 84
type: A, layer: 3, pos: 81
type: A, layer: 3, pos: 873
type: A, layer: 3, pos: 665
type: A, layer: 3, pos: 80
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 324
type: A, layer: 3, pos: 109
type: A, layer: 3, pos: 1007
type: A, layer: 3, pos: 844
type: A, layer: 3, pos: 82
type: A, layer: 3, pos: 663
type: A, layer: 3, pos: 589
type: A, layer: 3, pos: 696
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 209
type: A, layer: 3, pos: 94
type: A, layer: 3, pos: 296
type: A, layer: 3, pos: 990
type: A, layer: 3, pos: 251
type: A, layer: 3, pos: 885
type: A, layer: 3, pos: 989
type: A, layer: 3, pos: 261
type: A, layer: 3, pos: 104
type: A, layer: 3, pos: 681
type: A, layer: 3, pos: 34
type: A, layer: 3, pos: 358
type: A, layer: 3, pos: 853
type: A, layer: 3, pos: 304
type: A, layer: 3, pos: 280
type: A, layer: 3, pos: 127
type: A, layer: 3, pos: 628
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 249
type: A, layer: 3, pos: 674
type: A, layer: 3, pos: 833
type: A, layer: 3, pos: 998
type: A, layer: 3, pos: 986
type: A, layer: 3, pos: 610
type: A, layer: 3, pos: 207
type: A, layer: 3, pos: 847
type: A, layer: 3, pos: 123
type: A, layer: 3, pos: 53
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 597
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 672
type: A, layer: 3, pos: 270
type: A, layer: 3, pos: 590
type: A, layer: 3, pos: 202
type: A, layer: 3, pos: 250
type: A, layer: 3, pos: 282
type: A, layer: 3, pos: 368
type: A, layer: 3, pos: 596
type: A, layer: 3, pos: 1001
type: A, layer: 3, pos: 364
type: A, layer: 3, pos: 321
type: A, layer: 3, pos: 126
type: A, layer: 3, pos: 834
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 362
type: A, layer: 3, pos: 970
type: A, layer: 3, pos: 275
type: A, layer: 3, pos: 1002
type: A, layer: 3, pos: 617
type: A, layer: 3, pos: 973
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 630
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 352
type: A, layer: 3, pos: 205
type: A, layer: 3, pos: 97
type: A, layer: 3, pos: 24
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 1006
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 60
type: A, layer: 3, pos: 725
type: A, layer: 3, pos: 606
type: A, layer: 3, pos: 52
type: A, layer: 3, pos: 112
type: A, layer: 3, pos: 276
type: A, layer: 3, pos: 845
type: A, layer: 3, pos: 602
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 119
type: A, layer: 3, pos: 588
type: A, layer: 3, pos: 215
type: A, layer: 3, pos: 320
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 217
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 593
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 871
type: A, layer: 3, pos: 301
type: A, layer: 3, pos: 979
type: A, layer: 3, pos: 594
type: A, layer: 3, pos: 629
type: A, layer: 3, pos: 1022
type: A, layer: 3, pos: 866
type: A, layer: 3, pos: 654
type: A, layer: 3, pos: 1008
type: A, layer: 3, pos: 89
type: A, layer: 3, pos: 114
type: A, layer: 3, pos: 87
type: A, layer: 3, pos: 862
type: A, layer: 3, pos: 587
type: A, layer: 3, pos: 580
type: A, layer: 3, pos: 88
type: A, layer: 3, pos: 966
type: A, layer: 3, pos: 210
type: A, layer: 3, pos: 659
type: A, layer: 3, pos: 960
type: A, layer: 3, pos: 852
type: A, layer: 3, pos: 694
type: A, layer: 3, pos: 692
type: A, layer: 3, pos: 371
type: A, layer: 3, pos: 1016
type: A, layer: 3, pos: 247
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 591
type: A, layer: 3, pos: 599
type: A, layer: 3, pos: 103
type: A, layer: 3, pos: 288
type: A, layer: 3, pos: 351
type: A, layer: 3, pos: 341
type: A, layer: 3, pos: 631
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 267
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 367
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 256
type: A, layer: 3, pos: 627
type: A, layer: 3, pos: 266
type: A, layer: 3, pos: 271
type: A, layer: 3, pos: 413
type: A, layer: 3, pos: 653
type: A, layer: 3, pos: 234
type: A, layer: 3, pos: 652
type: A, layer: 3, pos: 586
type: A, layer: 3, pos: 585
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 682
type: A, layer: 3, pos: 105
type: A, layer: 3, pos: 601
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 623
type: A, layer: 3, pos: 110
type: A, layer: 3, pos: 968
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 23
type: A, layer: 3, pos: 701
type: A, layer: 3, pos: 691
type: A, layer: 3, pos: 9
type: A, layer: 3, pos: 1011
type: A, layer: 3, pos: 685
type: A, layer: 3, pos: 125
type: A, layer: 3, pos: 639
type: A, layer: 3, pos: 74
type: A, layer: 3, pos: 650
type: A, layer: 3, pos: 687
type: A, layer: 3, pos: 680
type: A, layer: 3, pos: 121
type: A, layer: 3, pos: 312
type: A, layer: 3, pos: 230
type: A, layer: 3, pos: 269
type: A, layer: 3, pos: 285
type: A, layer: 3, pos: 303
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 618
type: A, layer: 3, pos: 609
type: A, layer: 3, pos: 622
type: A, layer: 3, pos: 636
type: A, layer: 3, pos: 405
type: A, layer: 3, pos: 669
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 607
type: A, layer: 3, pos: 243
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 626

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 237

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1389150, upper bound: 14.6366811
time: 76.79 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1389150, upper bound: 14.6366811
time: 68.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 146.71 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 146.71
Output dim: 4, lower bound: -14.1787474, upper bound: 14.4825294
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 146.71
Output dim: 4, lower bound: -14.1787474, upper bound: 14.4825294
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 146.71
Output dim: 4, lower bound: -14.1787474, upper bound: 14.6499224
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 146.71
Output dim: 4, lower bound: -14.1787474, upper bound: 14.6499224
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 146.71
Output dim: 4, lower bound: -14.1389150, upper bound: 14.5140414
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 146.71
Output dim: 4, lower bound: -14.1389150, upper bound: 14.5140414
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 146.71
Output dim: 4, lower bound: -14.1389150, upper bound: 14.6366811
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 146.71
Output dim: 4, lower bound: -14.1389150, upper bound: 14.6366811

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -37.4292336, -0.3262672, -37.4591217, -0.3054848, -37.1237488, 37.1328545
1: -17.5082302, 10.4491310, -17.5411053, 10.4592743, -27.9675045, 27.9902363
2: -14.1709681, 9.9897842, -14.2401886, 10.0653934, -24.2363625, 24.2299728
3: -14.7313843, 14.0067825, -14.7570610, 14.0394516, -28.7708359, 28.7638435
4: -14.7299709, 14.6001139, -14.8640404, 14.7308578, -29.4608288, 29.4641533
5: -14.0487680, 15.1413803, -14.0741186, 15.1690273, -29.2177963, 29.2154999
6: -20.6428776, 10.0801964, -20.7892113, 10.1958399, -30.8387184, 30.8694077
7: -17.2240639, 16.5009251, -17.2547626, 16.5100956, -33.3633575, 33.3845444
8: -15.9681892, 19.0010109, -16.0940781, 19.1316376, -35.0559235, 35.0334511
9: -14.9781361, 13.5573778, -15.1061363, 13.6566067, -28.4571381, 28.4774761
10: -23.1846085, 16.7012901, -23.4718285, 16.9012146, -40.0858231, 40.1731186
11: -26.0395660, 9.9199190, -26.1731873, 10.0459356, -36.0855026, 36.0931053
12: -24.0548401, 11.6207523, -24.1965122, 11.7400932, -35.7949333, 35.8172646
13: -22.1378708, 18.3306465, -22.1631432, 18.3769817, -40.5148544, 40.4937897
14: -47.7699928, -0.6094933, -47.8079605, -0.6163635, -46.9959717, 47.0452881
15: -19.4025116, 10.2351313, -19.5022659, 10.2851801, -29.6876907, 29.7373962
16: -24.7140636, 12.9076223, -24.8918247, 13.0618105, -37.3239326, 37.3471985
17: -43.8686485, 12.0757484, -43.9021378, 12.1261940, -54.6997528, 54.6821823
18: -20.3505554, 12.3999672, -20.4007149, 12.4271250, -32.7776794, 32.8006821
19: -17.8251038, 4.1734266, -17.8710804, 4.1922655, -22.0173683, 22.0445061
20: -15.2102795, 8.4161997, -15.2370338, 8.4259892, -23.6362686, 23.6532326
21: -25.7908783, 3.6278143, -25.8444595, 3.6463690, -29.4372482, 29.4722748
22: -32.7880707, -1.0064554, -32.8742828, -0.9564381, -30.4779739, 30.5435829
23: -17.8546066, 8.8068609, -17.8895569, 8.8431454, -26.6977520, 26.6964188
24: -25.1249466, 7.2080355, -25.2095375, 7.3231902, -30.9623909, 30.9463348
25: -18.2304268, 10.7397308, -18.2787361, 10.7600718, -28.9904976, 29.0184669
26: -23.6105137, 14.7210722, -23.6569386, 14.7381878, -38.3487015, 38.3780098
27: -26.1163960, 6.6131382, -26.2034798, 6.6696482, -31.7921448, 31.8177853
28: -17.2703876, 10.5765276, -17.2875042, 10.5955524, -27.6671219, 27.6688194
29: -40.0786858, -5.4668102, -40.1099930, -5.4542046, -33.6625671, 33.6968613
30: -20.8210526, 12.2272472, -20.8440742, 12.2525578, -33.0736084, 33.0713196
31: -23.6022835, 6.9450102, -23.6573601, 6.9539776, -30.5562611, 30.6023712
32: -27.5064526, 4.1734467, -27.6057320, 4.2521086, -30.8615799, 30.8933754
33: -30.3147297, 14.4939995, -30.3763962, 14.5509987, -43.9391403, 43.9321060
34: -25.8629570, 9.8427124, -25.8894958, 9.8749733, -35.7379303, 35.7322083
35: -27.5442600, 10.8446140, -27.6151314, 10.9314976, -38.0389252, 38.0224915
36: -27.1000900, 10.8478184, -27.1340008, 10.8704014, -37.5274353, 37.5385017
37: -37.1214066, 9.5747423, -37.1584206, 9.5938206, -45.4129257, 45.4317207
38: -29.5559349, 13.9281559, -29.6050987, 13.9584064, -43.5143433, 43.5332565
39: -38.2440567, 11.5568314, -38.3314247, 11.5991774, -49.1991959, 49.2446861
40: -30.3092899, 9.7549038, -30.3191566, 9.7751236, -38.4389229, 38.4296684
41: -22.2438545, 9.3847561, -22.3373051, 9.4495621, -31.6934166, 31.7220612
42: -16.1541386, 7.1536593, -16.3577309, 7.3169179, -23.1748638, 23.2139950

Time for backsubstitution: 0.94 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 292
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 229
type: B, layer: 3, pos: 868
type: B, layer: 3, pos: 355
type: B, layer: 3, pos: 236
type: B, layer: 3, pos: 357
type: B, layer: 3, pos: 363
type: B, layer: 3, pos: 348
type: B, layer: 3, pos: 284
type: B, layer: 3, pos: 869
type: B, layer: 3, pos: 997
type: B, layer: 3, pos: 887
type: B, layer: 3, pos: 377
type: B, layer: 3, pos: 353
type: B, layer: 3, pos: 369
type: B, layer: 3, pos: 375
type: B, layer: 3, pos: 875
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 293
type: B, layer: 3, pos: 988
type: B, layer: 3, pos: 999
type: B, layer: 3, pos: 881
type: B, layer: 3, pos: 291
type: B, layer: 3, pos: 378
type: B, layer: 3, pos: 991
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 283
type: B, layer: 3, pos: 305
type: B, layer: 3, pos: 996
type: B, layer: 3, pos: 289
type: B, layer: 3, pos: 383
type: B, layer: 3, pos: 993
type: B, layer: 3, pos: 380
type: B, layer: 3, pos: 1009
type: B, layer: 3, pos: 893
type: B, layer: 3, pos: 331
type: B, layer: 3, pos: 877
type: B, layer: 3, pos: 339
type: B, layer: 3, pos: 361
type: B, layer: 3, pos: 849
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 338
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 1015
type: B, layer: 3, pos: 972
type: B, layer: 3, pos: 850
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 231
type: B, layer: 3, pos: 889
type: B, layer: 3, pos: 865
type: B, layer: 3, pos: 843
type: B, layer: 3, pos: 684
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 300
type: B, layer: 3, pos: 859
type: B, layer: 3, pos: 895
type: B, layer: 3, pos: 882
type: B, layer: 3, pos: 347
type: B, layer: 3, pos: 689
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 841
type: B, layer: 3, pos: 1023
type: B, layer: 3, pos: 379
type: B, layer: 3, pos: 382
type: B, layer: 3, pos: 644
type: B, layer: 3, pos: 860
type: B, layer: 3, pos: 695
type: B, layer: 3, pos: 223
type: B, layer: 3, pos: 239
type: B, layer: 3, pos: 346
type: B, layer: 3, pos: 329
type: B, layer: 3, pos: 306
type: B, layer: 3, pos: 265
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 1003
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 313
type: B, layer: 3, pos: 273
type: B, layer: 3, pos: 334
type: B, layer: 3, pos: 314
type: B, layer: 3, pos: 85
type: B, layer: 3, pos: 978
type: B, layer: 3, pos: 874
type: B, layer: 3, pos: 1005
type: B, layer: 3, pos: 58
type: B, layer: 3, pos: 1021
type: B, layer: 3, pos: 846
type: B, layer: 3, pos: 69
type: B, layer: 3, pos: 884
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 647
type: B, layer: 3, pos: 1017
type: B, layer: 3, pos: 699
type: B, layer: 3, pos: 274
type: B, layer: 3, pos: 977
type: B, layer: 3, pos: 299
type: B, layer: 3, pos: 894
type: B, layer: 3, pos: 974
type: B, layer: 3, pos: 995
type: B, layer: 3, pos: 370
type: B, layer: 3, pos: 851
type: B, layer: 3, pos: 646
type: B, layer: 3, pos: 876
type: B, layer: 3, pos: 698
type: B, layer: 3, pos: 667
type: B, layer: 3, pos: 260
type: B, layer: 3, pos: 381
type: B, layer: 3, pos: 235
type: B, layer: 3, pos: 673
type: B, layer: 3, pos: 1019
type: B, layer: 3, pos: 316
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 419
type: B, layer: 3, pos: 867
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 980
type: B, layer: 3, pos: 967
type: B, layer: 3, pos: 319
type: B, layer: 3, pos: 315
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 258
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 376
type: B, layer: 3, pos: 259
type: B, layer: 3, pos: 883
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 700
type: B, layer: 3, pos: 836
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 1020
type: B, layer: 3, pos: 842
type: B, layer: 3, pos: 61
type: B, layer: 3, pos: 1018
type: B, layer: 3, pos: 1014
type: B, layer: 3, pos: 336
type: B, layer: 3, pos: 688
type: B, layer: 3, pos: 56
type: B, layer: 3, pos: 272
type: B, layer: 3, pos: 1010
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 656
type: B, layer: 3, pos: 51
type: B, layer: 3, pos: 345
type: B, layer: 3, pos: 340
type: B, layer: 3, pos: 975
type: B, layer: 3, pos: 657
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 201
type: B, layer: 3, pos: 645
type: B, layer: 3, pos: 360
type: B, layer: 3, pos: 690
type: B, layer: 3, pos: 649
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 220
type: B, layer: 3, pos: 683
type: B, layer: 3, pos: 335
type: B, layer: 3, pos: 62
type: B, layer: 3, pos: 703
type: B, layer: 3, pos: 102
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 870
type: B, layer: 3, pos: 344
type: B, layer: 3, pos: 68
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 337
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 111
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 404
type: B, layer: 3, pos: 651
type: B, layer: 3, pos: 1013
type: B, layer: 3, pos: 349
type: B, layer: 3, pos: 858
type: B, layer: 3, pos: 981
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 242
type: B, layer: 3, pos: 861
type: B, layer: 3, pos: 1004
type: B, layer: 3, pos: 987
type: B, layer: 3, pos: 325
type: B, layer: 3, pos: 279
type: B, layer: 3, pos: 113
type: B, layer: 3, pos: 281
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 78
type: B, layer: 3, pos: 658
type: B, layer: 3, pos: 57
type: B, layer: 3, pos: 54
type: B, layer: 3, pos: 297
type: B, layer: 3, pos: 203
type: B, layer: 3, pos: 835
type: B, layer: 3, pos: 971
type: B, layer: 3, pos: 420
type: B, layer: 3, pos: 63
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 55
type: B, layer: 3, pos: 879
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 595
type: B, layer: 3, pos: 969
type: B, layer: 3, pos: 263
type: B, layer: 3, pos: 702
type: B, layer: 3, pos: 642
type: B, layer: 3, pos: 318
type: B, layer: 3, pos: 863
type: B, layer: 3, pos: 983
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 328
type: B, layer: 3, pos: 257
type: B, layer: 3, pos: 50
type: B, layer: 3, pos: 343
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 965
type: B, layer: 3, pos: 365
type: B, layer: 3, pos: 855
type: B, layer: 3, pos: 86
type: B, layer: 3, pos: 664
type: B, layer: 3, pos: 428
type: B, layer: 3, pos: 246
type: B, layer: 3, pos: 354
type: B, layer: 3, pos: 252
type: B, layer: 3, pos: 598
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 643
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 238
type: B, layer: 3, pos: 333
type: B, layer: 3, pos: 1012
type: B, layer: 3, pos: 982
type: B, layer: 3, pos: 264
type: B, layer: 3, pos: 124
type: B, layer: 3, pos: 262
type: B, layer: 3, pos: 648
type: B, layer: 3, pos: 641
type: B, layer: 3, pos: 985
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 77
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 857
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 372
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 666
type: B, layer: 3, pos: 322
type: B, layer: 3, pos: 84
type: B, layer: 3, pos: 873
type: B, layer: 3, pos: 665
type: B, layer: 3, pos: 81
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 324
type: B, layer: 3, pos: 109
type: B, layer: 3, pos: 844
type: B, layer: 3, pos: 1007
type: B, layer: 3, pos: 82
type: B, layer: 3, pos: 589
type: B, layer: 3, pos: 663
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 696
type: B, layer: 3, pos: 209
type: B, layer: 3, pos: 94
type: B, layer: 3, pos: 296
type: B, layer: 3, pos: 989
type: B, layer: 3, pos: 885
type: B, layer: 3, pos: 990
type: B, layer: 3, pos: 251
type: B, layer: 3, pos: 681
type: B, layer: 3, pos: 104
type: B, layer: 3, pos: 261
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 358
type: B, layer: 3, pos: 127
type: B, layer: 3, pos: 421
type: B, layer: 3, pos: 304
type: B, layer: 3, pos: 280
type: B, layer: 3, pos: 853
type: B, layer: 3, pos: 628
type: B, layer: 3, pos: 249
type: B, layer: 3, pos: 833
type: B, layer: 3, pos: 674
type: B, layer: 3, pos: 207
type: B, layer: 3, pos: 610
type: B, layer: 3, pos: 986
type: B, layer: 3, pos: 123
type: B, layer: 3, pos: 998
type: B, layer: 3, pos: 847
type: B, layer: 3, pos: 53
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 597
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 672
type: B, layer: 3, pos: 270
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 590
type: B, layer: 3, pos: 202
type: B, layer: 3, pos: 282
type: B, layer: 3, pos: 368
type: B, layer: 3, pos: 596
type: B, layer: 3, pos: 1001
type: B, layer: 3, pos: 364
type: B, layer: 3, pos: 321
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 126
type: B, layer: 3, pos: 834
type: B, layer: 3, pos: 970
type: B, layer: 3, pos: 362
type: B, layer: 3, pos: 275
type: B, layer: 3, pos: 973
type: B, layer: 3, pos: 1002
type: B, layer: 3, pos: 617
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 630
type: B, layer: 3, pos: 352
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 205
type: B, layer: 3, pos: 97
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 1006
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 276
type: B, layer: 3, pos: 606
type: B, layer: 3, pos: 52
type: B, layer: 3, pos: 112
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 845
type: B, layer: 3, pos: 602
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 588
type: B, layer: 3, pos: 119
type: B, layer: 3, pos: 320
type: B, layer: 3, pos: 215
type: B, layer: 3, pos: 217
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 593
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 871
type: B, layer: 3, pos: 979
type: B, layer: 3, pos: 594
type: B, layer: 3, pos: 629
type: B, layer: 3, pos: 301
type: B, layer: 3, pos: 1022
type: B, layer: 3, pos: 866
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1008
type: B, layer: 3, pos: 89
type: B, layer: 3, pos: 114
type: B, layer: 3, pos: 87
type: B, layer: 3, pos: 862
type: B, layer: 3, pos: 587
type: B, layer: 3, pos: 580
type: B, layer: 3, pos: 88
type: B, layer: 3, pos: 966
type: B, layer: 3, pos: 210
type: B, layer: 3, pos: 852
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 960
type: B, layer: 3, pos: 694
type: B, layer: 3, pos: 692
type: B, layer: 3, pos: 371
type: B, layer: 3, pos: 1016
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 247
type: B, layer: 3, pos: 591
type: B, layer: 3, pos: 599
type: B, layer: 3, pos: 103
type: B, layer: 3, pos: 288
type: B, layer: 3, pos: 351
type: B, layer: 3, pos: 631
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 341
type: B, layer: 3, pos: 267
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 367
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 256
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 271
type: B, layer: 3, pos: 266
type: B, layer: 3, pos: 413
type: B, layer: 3, pos: 653
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 586
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 585
type: B, layer: 3, pos: 682
type: B, layer: 3, pos: 105
type: B, layer: 3, pos: 601
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 623
type: B, layer: 3, pos: 110
type: B, layer: 3, pos: 968
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 701
type: B, layer: 3, pos: 691
type: B, layer: 3, pos: 1011
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 685
type: B, layer: 3, pos: 125
type: B, layer: 3, pos: 639
type: B, layer: 3, pos: 650
type: B, layer: 3, pos: 74
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 687
type: B, layer: 3, pos: 680
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 269
type: B, layer: 3, pos: 285
type: B, layer: 3, pos: 303
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 618
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 609
type: B, layer: 3, pos: 622
type: B, layer: 3, pos: 636
type: B, layer: 3, pos: 405
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 607
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 243
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 626

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 237

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.3605035, upper bound: 14.3620504
time: 58.40 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.3605035, upper bound: 14.4673187
time: 54.31 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -37.4530563, -0.3052864, -37.4669876, -0.3020725, -37.1509857, 37.1617012
1: -17.5303917, 10.4613171, -17.5487099, 10.4636812, -27.9940720, 28.0100269
2: -14.2540350, 10.0675869, -14.2686090, 10.0692825, -24.3233185, 24.3361969
3: -14.7544823, 14.0430660, -14.7642965, 14.0461807, -28.8006630, 28.8073616
4: -14.8945980, 14.7326374, -14.9165478, 14.7348928, -29.6294899, 29.6491852
5: -14.0737123, 15.1703730, -14.0813055, 15.1727810, -29.2464943, 29.2516785
6: -20.7922306, 10.2384501, -20.7970886, 10.2498016, -31.0420322, 31.0057564
7: -17.2514038, 16.5103817, -17.2614212, 16.5122204, -33.3939514, 33.4025726
8: -16.1215801, 19.1342697, -16.1436195, 19.1386909, -35.2178268, 35.2578392
9: -15.1138496, 13.6958485, -15.1174221, 13.7033672, -28.6256981, 28.5883846
10: -23.4849510, 16.9898434, -23.4897976, 17.0013580, -40.4863091, 40.4796410
11: -26.1777611, 10.0649881, -26.1838379, 10.0953121, -36.2730713, 36.2488251
12: -24.1969604, 11.7715263, -24.2008266, 11.7902298, -35.9871902, 35.9723511
13: -22.1645489, 18.3773098, -22.1673698, 18.3897133, -40.5542603, 40.5446777
14: -47.7863045, -0.6132393, -47.8105049, -0.6094704, -47.0209198, 47.0389786
15: -19.5009003, 10.2899303, -19.5339470, 10.2946129, -29.7955132, 29.8238773
16: -24.9059219, 13.1060190, -24.9111481, 13.1315956, -37.5756760, 37.5180550
17: -43.9016380, 12.1317205, -43.9065285, 12.1429186, -54.7491913, 54.7425308
18: -20.3901711, 12.4370155, -20.4095173, 12.4400043, -32.8301773, 32.8465347
19: -17.8747234, 4.1938968, -17.8797188, 4.1993880, -22.0741119, 22.0736160
20: -15.2381334, 8.4254017, -15.2427378, 8.4295588, -23.6676922, 23.6681404
21: -25.8470707, 3.6466484, -25.8525181, 3.6531677, -29.5002384, 29.4991665
22: -32.8692017, -0.9507160, -32.8995705, -0.9481993, -30.5473251, 30.5788193
23: -17.8909073, 8.8444862, -17.8945236, 8.8549318, -26.7458382, 26.7390099
24: -25.2305870, 7.3236656, -25.2450314, 7.3249917, -30.9874229, 31.0835819
25: -18.2876987, 10.7604046, -18.2956696, 10.7635727, -29.0512714, 29.0560741
26: -23.6347618, 14.7477722, -23.6608009, 14.7506304, -38.3853912, 38.4085732
27: -26.2068024, 6.6721115, -26.2308273, 6.6737156, -31.8535385, 31.9098282
28: -17.2877083, 10.5963049, -17.2916069, 10.5986729, -27.6902008, 27.6901321
29: -40.1036301, -5.4516830, -40.1169548, -5.4493980, -33.7547455, 33.7005692
30: -20.8459930, 12.2460938, -20.8505325, 12.2574177, -33.1034088, 33.0966263
31: -23.6619797, 6.9528399, -23.6687889, 6.9563875, -30.6183662, 30.6216278
32: -27.6078110, 4.2811470, -27.6107674, 4.2887573, -31.0090027, 30.9765930
33: -30.3753853, 14.5532513, -30.3964329, 14.5572910, -44.0020676, 44.0302963
34: -25.8844738, 9.8836098, -25.8956490, 9.8865452, -35.7710190, 35.7792587
35: -27.6285133, 10.9369678, -27.6445274, 10.9399929, -38.1190872, 38.1413803
36: -27.1288624, 10.8741322, -27.1418800, 10.8763695, -37.5588608, 37.5702019
37: -37.1610107, 9.5953712, -37.1692734, 9.5979424, -45.4540024, 45.4603729
38: -29.5994568, 13.9656487, -29.6175671, 13.9680910, -43.5675468, 43.5832138
39: -38.3388672, 11.5995178, -38.3589897, 11.6015310, -49.2823410, 49.3110352
40: -30.3246536, 9.7321310, -30.3301392, 9.7684221, -38.4525337, 38.4256325
41: -22.3407955, 9.4625483, -22.3461590, 9.4758425, -31.8166389, 31.7921219
42: -16.3595657, 7.3821220, -16.3630638, 7.3935194, -23.4565258, 23.3174553

Time for backsubstitution: 1.03 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 292
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 229
type: B, layer: 3, pos: 868
type: B, layer: 3, pos: 355
type: B, layer: 3, pos: 236
type: B, layer: 3, pos: 357
type: B, layer: 3, pos: 363
type: B, layer: 3, pos: 348
type: B, layer: 3, pos: 284
type: B, layer: 3, pos: 869
type: B, layer: 3, pos: 997
type: B, layer: 3, pos: 887
type: B, layer: 3, pos: 377
type: B, layer: 3, pos: 353
type: B, layer: 3, pos: 369
type: B, layer: 3, pos: 375
type: B, layer: 3, pos: 875
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 293
type: B, layer: 3, pos: 988
type: B, layer: 3, pos: 999
type: B, layer: 3, pos: 881
type: B, layer: 3, pos: 291
type: B, layer: 3, pos: 378
type: B, layer: 3, pos: 991
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 283
type: B, layer: 3, pos: 305
type: B, layer: 3, pos: 996
type: B, layer: 3, pos: 289
type: B, layer: 3, pos: 383
type: B, layer: 3, pos: 993
type: B, layer: 3, pos: 380
type: B, layer: 3, pos: 1009
type: B, layer: 3, pos: 893
type: B, layer: 3, pos: 877
type: B, layer: 3, pos: 331
type: B, layer: 3, pos: 361
type: B, layer: 3, pos: 339
type: B, layer: 3, pos: 849
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 338
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 1015
type: B, layer: 3, pos: 972
type: B, layer: 3, pos: 850
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 231
type: B, layer: 3, pos: 889
type: B, layer: 3, pos: 865
type: B, layer: 3, pos: 843
type: B, layer: 3, pos: 684
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 300
type: B, layer: 3, pos: 859
type: B, layer: 3, pos: 895
type: B, layer: 3, pos: 882
type: B, layer: 3, pos: 347
type: B, layer: 3, pos: 689
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 841
type: B, layer: 3, pos: 1023
type: B, layer: 3, pos: 379
type: B, layer: 3, pos: 382
type: B, layer: 3, pos: 644
type: B, layer: 3, pos: 860
type: B, layer: 3, pos: 695
type: B, layer: 3, pos: 223
type: B, layer: 3, pos: 239
type: B, layer: 3, pos: 346
type: B, layer: 3, pos: 329
type: B, layer: 3, pos: 306
type: B, layer: 3, pos: 265
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 1003
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 313
type: B, layer: 3, pos: 273
type: B, layer: 3, pos: 334
type: B, layer: 3, pos: 314
type: B, layer: 3, pos: 85
type: B, layer: 3, pos: 978
type: B, layer: 3, pos: 874
type: B, layer: 3, pos: 1005
type: B, layer: 3, pos: 58
type: B, layer: 3, pos: 1021
type: B, layer: 3, pos: 846
type: B, layer: 3, pos: 69
type: B, layer: 3, pos: 884
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 647
type: B, layer: 3, pos: 1017
type: B, layer: 3, pos: 699
type: B, layer: 3, pos: 274
type: B, layer: 3, pos: 977
type: B, layer: 3, pos: 299
type: B, layer: 3, pos: 894
type: B, layer: 3, pos: 974
type: B, layer: 3, pos: 995
type: B, layer: 3, pos: 370
type: B, layer: 3, pos: 851
type: B, layer: 3, pos: 646
type: B, layer: 3, pos: 698
type: B, layer: 3, pos: 876
type: B, layer: 3, pos: 667
type: B, layer: 3, pos: 260
type: B, layer: 3, pos: 381
type: B, layer: 3, pos: 235
type: B, layer: 3, pos: 673
type: B, layer: 3, pos: 1019
type: B, layer: 3, pos: 316
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 419
type: B, layer: 3, pos: 867
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 980
type: B, layer: 3, pos: 967
type: B, layer: 3, pos: 319
type: B, layer: 3, pos: 315
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 258
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 376
type: B, layer: 3, pos: 259
type: B, layer: 3, pos: 883
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 700
type: B, layer: 3, pos: 836
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 1020
type: B, layer: 3, pos: 842
type: B, layer: 3, pos: 61
type: B, layer: 3, pos: 1018
type: B, layer: 3, pos: 1014
type: B, layer: 3, pos: 336
type: B, layer: 3, pos: 688
type: B, layer: 3, pos: 56
type: B, layer: 3, pos: 272
type: B, layer: 3, pos: 1010
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 656
type: B, layer: 3, pos: 51
type: B, layer: 3, pos: 345
type: B, layer: 3, pos: 340
type: B, layer: 3, pos: 975
type: B, layer: 3, pos: 657
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 201
type: B, layer: 3, pos: 645
type: B, layer: 3, pos: 360
type: B, layer: 3, pos: 690
type: B, layer: 3, pos: 649
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 683
type: B, layer: 3, pos: 220
type: B, layer: 3, pos: 335
type: B, layer: 3, pos: 62
type: B, layer: 3, pos: 703
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 102
type: B, layer: 3, pos: 870
type: B, layer: 3, pos: 344
type: B, layer: 3, pos: 68
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 337
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 111
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 404
type: B, layer: 3, pos: 651
type: B, layer: 3, pos: 1013
type: B, layer: 3, pos: 349
type: B, layer: 3, pos: 858
type: B, layer: 3, pos: 981
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 242
type: B, layer: 3, pos: 861
type: B, layer: 3, pos: 1004
type: B, layer: 3, pos: 987
type: B, layer: 3, pos: 279
type: B, layer: 3, pos: 325
type: B, layer: 3, pos: 113
type: B, layer: 3, pos: 281
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 78
type: B, layer: 3, pos: 658
type: B, layer: 3, pos: 57
type: B, layer: 3, pos: 54
type: B, layer: 3, pos: 297
type: B, layer: 3, pos: 203
type: B, layer: 3, pos: 835
type: B, layer: 3, pos: 971
type: B, layer: 3, pos: 420
type: B, layer: 3, pos: 63
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 55
type: B, layer: 3, pos: 879
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 595
type: B, layer: 3, pos: 969
type: B, layer: 3, pos: 263
type: B, layer: 3, pos: 642
type: B, layer: 3, pos: 702
type: B, layer: 3, pos: 318
type: B, layer: 3, pos: 863
type: B, layer: 3, pos: 983
type: B, layer: 3, pos: 328
type: B, layer: 3, pos: 257
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 50
type: B, layer: 3, pos: 343
type: B, layer: 3, pos: 965
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 365
type: B, layer: 3, pos: 855
type: B, layer: 3, pos: 428
type: B, layer: 3, pos: 664
type: B, layer: 3, pos: 86
type: B, layer: 3, pos: 246
type: B, layer: 3, pos: 354
type: B, layer: 3, pos: 252
type: B, layer: 3, pos: 598
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 643
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 238
type: B, layer: 3, pos: 333
type: B, layer: 3, pos: 1012
type: B, layer: 3, pos: 264
type: B, layer: 3, pos: 982
type: B, layer: 3, pos: 124
type: B, layer: 3, pos: 262
type: B, layer: 3, pos: 648
type: B, layer: 3, pos: 641
type: B, layer: 3, pos: 985
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 77
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 857
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 372
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 666
type: B, layer: 3, pos: 322
type: B, layer: 3, pos: 84
type: B, layer: 3, pos: 873
type: B, layer: 3, pos: 81
type: B, layer: 3, pos: 665
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 324
type: B, layer: 3, pos: 109
type: B, layer: 3, pos: 844
type: B, layer: 3, pos: 1007
type: B, layer: 3, pos: 82
type: B, layer: 3, pos: 589
type: B, layer: 3, pos: 663
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 696
type: B, layer: 3, pos: 209
type: B, layer: 3, pos: 94
type: B, layer: 3, pos: 296
type: B, layer: 3, pos: 989
type: B, layer: 3, pos: 885
type: B, layer: 3, pos: 251
type: B, layer: 3, pos: 990
type: B, layer: 3, pos: 104
type: B, layer: 3, pos: 681
type: B, layer: 3, pos: 261
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 358
type: B, layer: 3, pos: 127
type: B, layer: 3, pos: 304
type: B, layer: 3, pos: 280
type: B, layer: 3, pos: 628
type: B, layer: 3, pos: 421
type: B, layer: 3, pos: 853
type: B, layer: 3, pos: 249
type: B, layer: 3, pos: 833
type: B, layer: 3, pos: 674
type: B, layer: 3, pos: 207
type: B, layer: 3, pos: 986
type: B, layer: 3, pos: 610
type: B, layer: 3, pos: 123
type: B, layer: 3, pos: 998
type: B, layer: 3, pos: 847
type: B, layer: 3, pos: 53
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 597
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 672
type: B, layer: 3, pos: 270
type: B, layer: 3, pos: 590
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 202
type: B, layer: 3, pos: 282
type: B, layer: 3, pos: 368
type: B, layer: 3, pos: 596
type: B, layer: 3, pos: 1001
type: B, layer: 3, pos: 321
type: B, layer: 3, pos: 364
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 834
type: B, layer: 3, pos: 126
type: B, layer: 3, pos: 970
type: B, layer: 3, pos: 362
type: B, layer: 3, pos: 275
type: B, layer: 3, pos: 973
type: B, layer: 3, pos: 1002
type: B, layer: 3, pos: 617
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 630
type: B, layer: 3, pos: 352
type: B, layer: 3, pos: 205
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 97
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 1006
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 276
type: B, layer: 3, pos: 606
type: B, layer: 3, pos: 52
type: B, layer: 3, pos: 112
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 845
type: B, layer: 3, pos: 602
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 588
type: B, layer: 3, pos: 119
type: B, layer: 3, pos: 320
type: B, layer: 3, pos: 215
type: B, layer: 3, pos: 217
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 593
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 871
type: B, layer: 3, pos: 979
type: B, layer: 3, pos: 594
type: B, layer: 3, pos: 629
type: B, layer: 3, pos: 301
type: B, layer: 3, pos: 1022
type: B, layer: 3, pos: 866
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1008
type: B, layer: 3, pos: 89
type: B, layer: 3, pos: 114
type: B, layer: 3, pos: 87
type: B, layer: 3, pos: 862
type: B, layer: 3, pos: 587
type: B, layer: 3, pos: 580
type: B, layer: 3, pos: 88
type: B, layer: 3, pos: 966
type: B, layer: 3, pos: 210
type: B, layer: 3, pos: 852
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 960
type: B, layer: 3, pos: 694
type: B, layer: 3, pos: 692
type: B, layer: 3, pos: 371
type: B, layer: 3, pos: 1016
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 247
type: B, layer: 3, pos: 591
type: B, layer: 3, pos: 599
type: B, layer: 3, pos: 103
type: B, layer: 3, pos: 351
type: B, layer: 3, pos: 288
type: B, layer: 3, pos: 631
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 341
type: B, layer: 3, pos: 267
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 367
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 256
type: B, layer: 3, pos: 271
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 266
type: B, layer: 3, pos: 413
type: B, layer: 3, pos: 653
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 586
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 585
type: B, layer: 3, pos: 682
type: B, layer: 3, pos: 105
type: B, layer: 3, pos: 601
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 623
type: B, layer: 3, pos: 110
type: B, layer: 3, pos: 968
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 701
type: B, layer: 3, pos: 691
type: B, layer: 3, pos: 1011
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 685
type: B, layer: 3, pos: 125
type: B, layer: 3, pos: 639
type: B, layer: 3, pos: 650
type: B, layer: 3, pos: 74
type: B, layer: 3, pos: 687
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 680
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 269
type: B, layer: 3, pos: 285
type: B, layer: 3, pos: 303
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 618
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 609
type: B, layer: 3, pos: 622
type: B, layer: 3, pos: 636
type: B, layer: 3, pos: 405
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 607
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 243
type: B, layer: 3, pos: 626

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 237

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.3605035, upper bound: 14.3620504
time: 57.82 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.3605035, upper bound: 14.4673187
time: 57.91 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -37.4292336, -0.3262672, -37.5717468, -0.1190395, -37.3101959, 37.2454796
1: -17.5082302, 10.4491310, -17.6085167, 10.5991268, -28.1073570, 28.0576477
2: -14.1709681, 9.9897842, -14.4205818, 10.4355049, -24.6064720, 24.4103661
3: -14.7313843, 14.0067825, -14.9194307, 14.3844986, -29.1158829, 28.9262123
4: -14.7299709, 14.6001139, -15.1261616, 15.2482014, -29.9781723, 29.7262764
5: -14.0487680, 15.1413803, -14.2462921, 15.5564632, -29.6052322, 29.3876724
6: -20.6428776, 10.0801964, -20.9641838, 10.2232609, -30.8661385, 31.0443802
7: -17.2240639, 16.5009251, -17.3878899, 16.7149620, -33.5733795, 33.5218620
8: -15.9681892, 19.0010109, -16.3111744, 19.5392342, -35.4641571, 35.2503128
9: -14.9781361, 13.5573778, -15.2108641, 13.7007027, -28.5024872, 28.5788040
10: -23.1846085, 16.7012901, -23.9720459, 17.1925964, -40.3772049, 40.6733360
11: -26.0395660, 9.9199190, -26.7462006, 10.3291969, -36.3687630, 36.6661186
12: -24.0548401, 11.6207523, -24.9791145, 12.1248417, -36.1796799, 36.5998688
13: -22.1378708, 18.3306465, -22.2509842, 18.4505997, -40.5884705, 40.5816307
14: -47.7699928, -0.6094933, -48.2812462, -0.3966770, -47.2136765, 47.5174904
15: -19.4025116, 10.2351313, -19.5713100, 10.4389124, -29.8414230, 29.8064423
16: -24.7140636, 12.9076223, -25.1079731, 13.0775785, -37.3545609, 37.5653458
17: -43.8686485, 12.0757484, -44.5339813, 12.4492607, -55.0231323, 55.3131790
18: -20.3505554, 12.3999672, -20.4511070, 12.4966764, -32.8472328, 32.8510742
19: -17.8251038, 4.1734266, -18.1492996, 4.2843523, -22.1094551, 22.3227272
20: -15.2102795, 8.4161997, -15.3727322, 8.4819546, -23.6922340, 23.7889328
21: -25.7908783, 3.6278143, -26.2082367, 3.7971272, -29.5880051, 29.8360519
22: -32.7880707, -1.0064554, -32.9963989, -0.8563404, -30.5947685, 30.6726341
23: -17.8546066, 8.8068609, -18.1549244, 8.9355650, -26.7901726, 26.9617844
24: -25.1249466, 7.2080355, -25.2368450, 7.3771935, -31.0150490, 30.9677315
25: -18.2304268, 10.7397308, -18.4059219, 10.8596420, -29.0900688, 29.1456528
26: -23.6105137, 14.7210722, -23.9739456, 14.9131374, -38.5236511, 38.6950188
27: -26.1163960, 6.6131382, -26.2566891, 6.7151709, -31.8316307, 31.8655243
28: -17.2703876, 10.5765276, -17.4452858, 10.6351357, -27.7072983, 27.8255539
29: -40.0786858, -5.4668102, -40.4464073, -5.2655640, -33.8544388, 34.0449982
30: -20.8210526, 12.2272472, -21.0250111, 12.3176250, -33.1386795, 33.2522583
31: -23.6022835, 6.9450102, -23.8461189, 7.0141349, -30.6164188, 30.7911301
32: -27.5064526, 4.1734467, -27.8142204, 4.3573856, -30.9663887, 31.1089478
33: -30.3147297, 14.4939995, -30.5583839, 14.8763590, -44.2653809, 44.1130524
34: -25.8629570, 9.8427124, -25.9964447, 10.0103292, -35.8732872, 35.8391571
35: -27.5442600, 10.8446140, -27.7376995, 11.1107368, -38.2322350, 38.1536064
36: -27.1000900, 10.8478184, -27.2275429, 10.8934174, -37.5554733, 37.6293716
37: -37.1214066, 9.5747423, -37.3332291, 9.6304016, -45.4641724, 45.6275101
38: -29.5559349, 13.9281559, -29.7362843, 14.0384769, -43.5944138, 43.6644402
39: -38.2440567, 11.5568314, -38.4966736, 11.7549076, -49.3461456, 49.4052048
40: -30.3092899, 9.7549038, -30.4348030, 9.8576508, -38.5122528, 38.5331116
41: -22.2438545, 9.3847561, -22.5147266, 9.4669914, -31.7108459, 31.8994827
42: -16.1541386, 7.1536593, -16.7699909, 7.5357118, -23.3960953, 23.6342373

Time for backsubstitution: 1.00 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 292
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 229
type: B, layer: 3, pos: 868
type: B, layer: 3, pos: 355
type: B, layer: 3, pos: 236
type: B, layer: 3, pos: 357
type: B, layer: 3, pos: 363
type: B, layer: 3, pos: 348
type: B, layer: 3, pos: 284
type: B, layer: 3, pos: 869
type: B, layer: 3, pos: 997
type: B, layer: 3, pos: 887
type: B, layer: 3, pos: 377
type: B, layer: 3, pos: 353
type: B, layer: 3, pos: 369
type: B, layer: 3, pos: 375
type: B, layer: 3, pos: 875
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 293
type: B, layer: 3, pos: 988
type: B, layer: 3, pos: 999
type: B, layer: 3, pos: 881
type: B, layer: 3, pos: 291
type: B, layer: 3, pos: 378
type: B, layer: 3, pos: 991
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 283
type: B, layer: 3, pos: 305
type: B, layer: 3, pos: 996
type: B, layer: 3, pos: 289
type: B, layer: 3, pos: 383
type: B, layer: 3, pos: 993
type: B, layer: 3, pos: 380
type: B, layer: 3, pos: 1009
type: B, layer: 3, pos: 893
type: B, layer: 3, pos: 331
type: B, layer: 3, pos: 877
type: B, layer: 3, pos: 361
type: B, layer: 3, pos: 339
type: B, layer: 3, pos: 849
type: B, layer: 3, pos: 338
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 1015
type: B, layer: 3, pos: 972
type: B, layer: 3, pos: 850
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 231
type: B, layer: 3, pos: 889
type: B, layer: 3, pos: 865
type: B, layer: 3, pos: 843
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 684
type: B, layer: 3, pos: 300
type: B, layer: 3, pos: 859
type: B, layer: 3, pos: 895
type: B, layer: 3, pos: 882
type: B, layer: 3, pos: 347
type: B, layer: 3, pos: 689
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 841
type: B, layer: 3, pos: 1023
type: B, layer: 3, pos: 379
type: B, layer: 3, pos: 382
type: B, layer: 3, pos: 644
type: B, layer: 3, pos: 860
type: B, layer: 3, pos: 695
type: B, layer: 3, pos: 223
type: B, layer: 3, pos: 239
type: B, layer: 3, pos: 346
type: B, layer: 3, pos: 329
type: B, layer: 3, pos: 265
type: B, layer: 3, pos: 306
type: B, layer: 3, pos: 1003
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 313
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 273
type: B, layer: 3, pos: 314
type: B, layer: 3, pos: 334
type: B, layer: 3, pos: 85
type: B, layer: 3, pos: 978
type: B, layer: 3, pos: 1005
type: B, layer: 3, pos: 874
type: B, layer: 3, pos: 58
type: B, layer: 3, pos: 1021
type: B, layer: 3, pos: 846
type: B, layer: 3, pos: 69
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 884
type: B, layer: 3, pos: 647
type: B, layer: 3, pos: 1017
type: B, layer: 3, pos: 699
type: B, layer: 3, pos: 274
type: B, layer: 3, pos: 977
type: B, layer: 3, pos: 299
type: B, layer: 3, pos: 894
type: B, layer: 3, pos: 995
type: B, layer: 3, pos: 974
type: B, layer: 3, pos: 851
type: B, layer: 3, pos: 370
type: B, layer: 3, pos: 646
type: B, layer: 3, pos: 876
type: B, layer: 3, pos: 698
type: B, layer: 3, pos: 667
type: B, layer: 3, pos: 260
type: B, layer: 3, pos: 381
type: B, layer: 3, pos: 673
type: B, layer: 3, pos: 235
type: B, layer: 3, pos: 1019
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 316
type: B, layer: 3, pos: 419
type: B, layer: 3, pos: 867
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 980
type: B, layer: 3, pos: 319
type: B, layer: 3, pos: 967
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 258
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 376
type: B, layer: 3, pos: 315
type: B, layer: 3, pos: 259
type: B, layer: 3, pos: 883
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 700
type: B, layer: 3, pos: 836
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 1020
type: B, layer: 3, pos: 842
type: B, layer: 3, pos: 61
type: B, layer: 3, pos: 1018
type: B, layer: 3, pos: 336
type: B, layer: 3, pos: 688
type: B, layer: 3, pos: 1014
type: B, layer: 3, pos: 56
type: B, layer: 3, pos: 272
type: B, layer: 3, pos: 1010
type: B, layer: 3, pos: 656
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 345
type: B, layer: 3, pos: 51
type: B, layer: 3, pos: 340
type: B, layer: 3, pos: 975
type: B, layer: 3, pos: 657
type: B, layer: 3, pos: 201
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 645
type: B, layer: 3, pos: 649
type: B, layer: 3, pos: 360
type: B, layer: 3, pos: 690
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 683
type: B, layer: 3, pos: 335
type: B, layer: 3, pos: 220
type: B, layer: 3, pos: 703
type: B, layer: 3, pos: 62
type: B, layer: 3, pos: 102
type: B, layer: 3, pos: 870
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 344
type: B, layer: 3, pos: 68
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 337
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 111
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 404
type: B, layer: 3, pos: 1013
type: B, layer: 3, pos: 651
type: B, layer: 3, pos: 981
type: B, layer: 3, pos: 858
type: B, layer: 3, pos: 349
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 861
type: B, layer: 3, pos: 242
type: B, layer: 3, pos: 1004
type: B, layer: 3, pos: 325
type: B, layer: 3, pos: 279
type: B, layer: 3, pos: 987
type: B, layer: 3, pos: 113
type: B, layer: 3, pos: 281
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 78
type: B, layer: 3, pos: 658
type: B, layer: 3, pos: 54
type: B, layer: 3, pos: 57
type: B, layer: 3, pos: 297
type: B, layer: 3, pos: 203
type: B, layer: 3, pos: 835
type: B, layer: 3, pos: 971
type: B, layer: 3, pos: 420
type: B, layer: 3, pos: 63
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 879
type: B, layer: 3, pos: 55
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 969
type: B, layer: 3, pos: 263
type: B, layer: 3, pos: 318
type: B, layer: 3, pos: 595
type: B, layer: 3, pos: 642
type: B, layer: 3, pos: 702
type: B, layer: 3, pos: 863
type: B, layer: 3, pos: 983
type: B, layer: 3, pos: 257
type: B, layer: 3, pos: 328
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 50
type: B, layer: 3, pos: 343
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 365
type: B, layer: 3, pos: 965
type: B, layer: 3, pos: 855
type: B, layer: 3, pos: 664
type: B, layer: 3, pos: 252
type: B, layer: 3, pos: 354
type: B, layer: 3, pos: 428
type: B, layer: 3, pos: 86
type: B, layer: 3, pos: 246
type: B, layer: 3, pos: 598
type: B, layer: 3, pos: 643
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 238
type: B, layer: 3, pos: 333
type: B, layer: 3, pos: 262
type: B, layer: 3, pos: 982
type: B, layer: 3, pos: 264
type: B, layer: 3, pos: 1012
type: B, layer: 3, pos: 124
type: B, layer: 3, pos: 648
type: B, layer: 3, pos: 641
type: B, layer: 3, pos: 985
type: B, layer: 3, pos: 857
type: B, layer: 3, pos: 77
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 372
type: B, layer: 3, pos: 666
type: B, layer: 3, pos: 322
type: B, layer: 3, pos: 81
type: B, layer: 3, pos: 84
type: B, layer: 3, pos: 873
type: B, layer: 3, pos: 665
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 324
type: B, layer: 3, pos: 109
type: B, layer: 3, pos: 1007
type: B, layer: 3, pos: 844
type: B, layer: 3, pos: 82
type: B, layer: 3, pos: 663
type: B, layer: 3, pos: 589
type: B, layer: 3, pos: 696
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 209
type: B, layer: 3, pos: 94
type: B, layer: 3, pos: 296
type: B, layer: 3, pos: 990
type: B, layer: 3, pos: 251
type: B, layer: 3, pos: 885
type: B, layer: 3, pos: 989
type: B, layer: 3, pos: 261
type: B, layer: 3, pos: 104
type: B, layer: 3, pos: 681
type: B, layer: 3, pos: 853
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 358
type: B, layer: 3, pos: 304
type: B, layer: 3, pos: 280
type: B, layer: 3, pos: 421
type: B, layer: 3, pos: 127
type: B, layer: 3, pos: 628
type: B, layer: 3, pos: 249
type: B, layer: 3, pos: 674
type: B, layer: 3, pos: 833
type: B, layer: 3, pos: 998
type: B, layer: 3, pos: 986
type: B, layer: 3, pos: 610
type: B, layer: 3, pos: 207
type: B, layer: 3, pos: 123
type: B, layer: 3, pos: 847
type: B, layer: 3, pos: 53
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 597
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 672
type: B, layer: 3, pos: 270
type: B, layer: 3, pos: 590
type: B, layer: 3, pos: 202
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 282
type: B, layer: 3, pos: 368
type: B, layer: 3, pos: 596
type: B, layer: 3, pos: 1001
type: B, layer: 3, pos: 364
type: B, layer: 3, pos: 321
type: B, layer: 3, pos: 126
type: B, layer: 3, pos: 834
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 362
type: B, layer: 3, pos: 970
type: B, layer: 3, pos: 275
type: B, layer: 3, pos: 1002
type: B, layer: 3, pos: 617
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 973
type: B, layer: 3, pos: 630
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 352
type: B, layer: 3, pos: 205
type: B, layer: 3, pos: 97
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 1006
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 606
type: B, layer: 3, pos: 52
type: B, layer: 3, pos: 276
type: B, layer: 3, pos: 112
type: B, layer: 3, pos: 845
type: B, layer: 3, pos: 602
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 119
type: B, layer: 3, pos: 215
type: B, layer: 3, pos: 588
type: B, layer: 3, pos: 320
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 217
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 593
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 871
type: B, layer: 3, pos: 301
type: B, layer: 3, pos: 979
type: B, layer: 3, pos: 594
type: B, layer: 3, pos: 629
type: B, layer: 3, pos: 1022
type: B, layer: 3, pos: 866
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1008
type: B, layer: 3, pos: 89
type: B, layer: 3, pos: 114
type: B, layer: 3, pos: 87
type: B, layer: 3, pos: 587
type: B, layer: 3, pos: 862
type: B, layer: 3, pos: 580
type: B, layer: 3, pos: 88
type: B, layer: 3, pos: 966
type: B, layer: 3, pos: 210
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 960
type: B, layer: 3, pos: 852
type: B, layer: 3, pos: 694
type: B, layer: 3, pos: 692
type: B, layer: 3, pos: 371
type: B, layer: 3, pos: 1016
type: B, layer: 3, pos: 247
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 591
type: B, layer: 3, pos: 599
type: B, layer: 3, pos: 103
type: B, layer: 3, pos: 288
type: B, layer: 3, pos: 351
type: B, layer: 3, pos: 341
type: B, layer: 3, pos: 631
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 267
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 367
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 256
type: B, layer: 3, pos: 266
type: B, layer: 3, pos: 271
type: B, layer: 3, pos: 413
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 653
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 586
type: B, layer: 3, pos: 585
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 682
type: B, layer: 3, pos: 105
type: B, layer: 3, pos: 601
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 623
type: B, layer: 3, pos: 110
type: B, layer: 3, pos: 968
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 701
type: B, layer: 3, pos: 691
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 1011
type: B, layer: 3, pos: 685
type: B, layer: 3, pos: 125
type: B, layer: 3, pos: 639
type: B, layer: 3, pos: 74
type: B, layer: 3, pos: 650
type: B, layer: 3, pos: 687
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 680
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 269
type: B, layer: 3, pos: 285
type: B, layer: 3, pos: 303
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 609
type: B, layer: 3, pos: 618
type: B, layer: 3, pos: 622
type: B, layer: 3, pos: 636
type: B, layer: 3, pos: 405
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 243
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 607
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 626

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 237

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1596706, upper bound: 14.5212905
time: 62.92 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1596706, upper bound: 14.6366804
time: 47.93 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -37.4530563, -0.3052864, -37.5809860, -0.1156368, -37.3374176, 37.2756996
1: -17.5303917, 10.4613171, -17.6196594, 10.6034050, -28.1337967, 28.0809765
2: -14.2540350, 10.0675869, -14.4486475, 10.4392662, -24.6933022, 24.5162354
3: -14.7544823, 14.0430660, -14.9279366, 14.3909569, -29.1454391, 28.9710026
4: -14.8945980, 14.7326374, -15.1784153, 15.2519703, -30.1465683, 29.9110527
5: -14.0737123, 15.1703730, -14.2539558, 15.5601444, -29.6338577, 29.4243279
6: -20.7922306, 10.2384501, -20.9720974, 10.2793941, -31.0716248, 31.2105484
7: -17.2514038, 16.5103817, -17.3954506, 16.7170830, -33.6038513, 33.5407410
8: -16.1215801, 19.1342697, -16.3608932, 19.5460815, -35.6258698, 35.4748840
9: -15.1138496, 13.6958485, -15.2220650, 13.7468815, -28.6732216, 28.6880760
10: -23.4849510, 16.9898434, -23.9891758, 17.2923393, -40.7772903, 40.9790192
11: -26.1777611, 10.0649881, -26.7565308, 10.3782673, -36.5560303, 36.8215179
12: -24.1969604, 11.7715263, -24.9833317, 12.1751394, -36.3721008, 36.7548599
13: -22.1645489, 18.3773098, -22.2552299, 18.4635239, -40.6280746, 40.6325378
14: -47.7863045, -0.6132393, -48.2837677, -0.3895340, -47.2388687, 47.5111427
15: -19.5009003, 10.2899303, -19.6061230, 10.4485245, -29.9494247, 29.8960533
16: -24.9059219, 13.1060190, -25.1271172, 13.1470222, -37.6080704, 37.7354012
17: -43.9016380, 12.1317205, -44.5383377, 12.4662628, -55.0728531, 55.3735199
18: -20.3901711, 12.4370155, -20.4603729, 12.5098648, -32.9000359, 32.8973885
19: -17.8747234, 4.1938968, -18.1575508, 4.2917461, -22.1664696, 22.3514481
20: -15.2381334, 8.4254017, -15.3783112, 8.4861803, -23.7243137, 23.8037128
21: -25.8470707, 3.6466484, -26.2160244, 3.8045201, -29.6515903, 29.8626728
22: -32.8692017, -0.9507160, -33.0216637, -0.8478050, -30.6678696, 30.7064896
23: -17.8909073, 8.8444862, -18.1596069, 8.9482460, -26.8391533, 27.0040932
24: -25.2305870, 7.3236656, -25.2722054, 7.3790770, -31.0392456, 31.1049194
25: -18.2876987, 10.7604046, -18.4217739, 10.8634243, -29.1511230, 29.1821785
26: -23.6347618, 14.7477722, -23.9773865, 14.9257889, -38.5605507, 38.7251587
27: -26.2068024, 6.6721115, -26.2841721, 6.7192240, -31.8927002, 31.9579525
28: -17.2877083, 10.5963049, -17.4492397, 10.6384287, -27.7331314, 27.8456993
29: -40.1036301, -5.4516830, -40.4520340, -5.2605515, -33.9467354, 34.0486832
30: -20.8459930, 12.2460938, -21.0314293, 12.3227940, -33.1687851, 33.2775230
31: -23.6619797, 6.9528399, -23.8573780, 7.0166903, -30.6786690, 30.8102188
32: -27.6078110, 4.2811470, -27.8191795, 4.3942890, -31.1134224, 31.1921196
33: -30.3753853, 14.5532513, -30.5784225, 14.8826962, -44.3283768, 44.2112808
34: -25.8844738, 9.8836098, -26.0028572, 10.0214005, -35.9058762, 35.8864670
35: -27.6285133, 10.9369678, -27.7671261, 11.1190071, -38.3121490, 38.2725334
36: -27.1288624, 10.8741322, -27.2356243, 10.8993835, -37.5870247, 37.6617279
37: -37.1610107, 9.5953712, -37.3436279, 9.6348019, -45.5055542, 45.6557465
38: -29.5994568, 13.9656487, -29.7505569, 14.0479794, -43.6474380, 43.7162056
39: -38.3388672, 11.5995178, -38.5248260, 11.7573090, -49.4291382, 49.4723206
40: -30.3246536, 9.7321310, -30.4461555, 9.8499193, -38.5250969, 38.5299873
41: -22.3407955, 9.4625483, -22.5235271, 9.4964552, -31.8372498, 31.9702892
42: -16.3595657, 7.3821220, -16.7751427, 7.6122546, -23.6771030, 23.7375469

Time for backsubstitution: 0.96 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 292
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 229
type: B, layer: 3, pos: 868
type: B, layer: 3, pos: 355
type: B, layer: 3, pos: 236
type: B, layer: 3, pos: 357
type: B, layer: 3, pos: 363
type: B, layer: 3, pos: 348
type: B, layer: 3, pos: 284
type: B, layer: 3, pos: 869
type: B, layer: 3, pos: 997
type: B, layer: 3, pos: 887
type: B, layer: 3, pos: 377
type: B, layer: 3, pos: 353
type: B, layer: 3, pos: 369
type: B, layer: 3, pos: 375
type: B, layer: 3, pos: 875
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 293
type: B, layer: 3, pos: 988
type: B, layer: 3, pos: 999
type: B, layer: 3, pos: 881
type: B, layer: 3, pos: 291
type: B, layer: 3, pos: 378
type: B, layer: 3, pos: 991
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 305
type: B, layer: 3, pos: 283
type: B, layer: 3, pos: 996
type: B, layer: 3, pos: 289
type: B, layer: 3, pos: 383
type: B, layer: 3, pos: 993
type: B, layer: 3, pos: 380
type: B, layer: 3, pos: 1009
type: B, layer: 3, pos: 893
type: B, layer: 3, pos: 331
type: B, layer: 3, pos: 877
type: B, layer: 3, pos: 361
type: B, layer: 3, pos: 339
type: B, layer: 3, pos: 849
type: B, layer: 3, pos: 338
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 1015
type: B, layer: 3, pos: 850
type: B, layer: 3, pos: 972
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 231
type: B, layer: 3, pos: 889
type: B, layer: 3, pos: 865
type: B, layer: 3, pos: 843
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 684
type: B, layer: 3, pos: 300
type: B, layer: 3, pos: 859
type: B, layer: 3, pos: 895
type: B, layer: 3, pos: 882
type: B, layer: 3, pos: 347
type: B, layer: 3, pos: 689
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 841
type: B, layer: 3, pos: 1023
type: B, layer: 3, pos: 379
type: B, layer: 3, pos: 382
type: B, layer: 3, pos: 644
type: B, layer: 3, pos: 860
type: B, layer: 3, pos: 695
type: B, layer: 3, pos: 223
type: B, layer: 3, pos: 239
type: B, layer: 3, pos: 346
type: B, layer: 3, pos: 329
type: B, layer: 3, pos: 265
type: B, layer: 3, pos: 306
type: B, layer: 3, pos: 1003
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 313
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 273
type: B, layer: 3, pos: 314
type: B, layer: 3, pos: 334
type: B, layer: 3, pos: 85
type: B, layer: 3, pos: 978
type: B, layer: 3, pos: 1005
type: B, layer: 3, pos: 874
type: B, layer: 3, pos: 58
type: B, layer: 3, pos: 1021
type: B, layer: 3, pos: 846
type: B, layer: 3, pos: 69
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 884
type: B, layer: 3, pos: 647
type: B, layer: 3, pos: 1017
type: B, layer: 3, pos: 699
type: B, layer: 3, pos: 274
type: B, layer: 3, pos: 977
type: B, layer: 3, pos: 299
type: B, layer: 3, pos: 894
type: B, layer: 3, pos: 995
type: B, layer: 3, pos: 974
type: B, layer: 3, pos: 370
type: B, layer: 3, pos: 851
type: B, layer: 3, pos: 646
type: B, layer: 3, pos: 698
type: B, layer: 3, pos: 876
type: B, layer: 3, pos: 667
type: B, layer: 3, pos: 260
type: B, layer: 3, pos: 381
type: B, layer: 3, pos: 673
type: B, layer: 3, pos: 235
type: B, layer: 3, pos: 1019
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 316
type: B, layer: 3, pos: 419
type: B, layer: 3, pos: 867
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 980
type: B, layer: 3, pos: 319
type: B, layer: 3, pos: 967
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 258
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 376
type: B, layer: 3, pos: 315
type: B, layer: 3, pos: 259
type: B, layer: 3, pos: 883
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 700
type: B, layer: 3, pos: 836
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 1020
type: B, layer: 3, pos: 842
type: B, layer: 3, pos: 61
type: B, layer: 3, pos: 1018
type: B, layer: 3, pos: 336
type: B, layer: 3, pos: 688
type: B, layer: 3, pos: 1014
type: B, layer: 3, pos: 56
type: B, layer: 3, pos: 272
type: B, layer: 3, pos: 1010
type: B, layer: 3, pos: 656
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 345
type: B, layer: 3, pos: 51
type: B, layer: 3, pos: 340
type: B, layer: 3, pos: 975
type: B, layer: 3, pos: 657
type: B, layer: 3, pos: 201
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 645
type: B, layer: 3, pos: 649
type: B, layer: 3, pos: 360
type: B, layer: 3, pos: 690
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 683
type: B, layer: 3, pos: 335
type: B, layer: 3, pos: 220
type: B, layer: 3, pos: 703
type: B, layer: 3, pos: 62
type: B, layer: 3, pos: 870
type: B, layer: 3, pos: 102
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 344
type: B, layer: 3, pos: 68
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 337
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 111
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 404
type: B, layer: 3, pos: 651
type: B, layer: 3, pos: 1013
type: B, layer: 3, pos: 981
type: B, layer: 3, pos: 349
type: B, layer: 3, pos: 858
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 861
type: B, layer: 3, pos: 242
type: B, layer: 3, pos: 1004
type: B, layer: 3, pos: 279
type: B, layer: 3, pos: 325
type: B, layer: 3, pos: 987
type: B, layer: 3, pos: 113
type: B, layer: 3, pos: 281
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 78
type: B, layer: 3, pos: 658
type: B, layer: 3, pos: 54
type: B, layer: 3, pos: 57
type: B, layer: 3, pos: 297
type: B, layer: 3, pos: 203
type: B, layer: 3, pos: 835
type: B, layer: 3, pos: 971
type: B, layer: 3, pos: 420
type: B, layer: 3, pos: 63
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 879
type: B, layer: 3, pos: 55
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 969
type: B, layer: 3, pos: 263
type: B, layer: 3, pos: 318
type: B, layer: 3, pos: 595
type: B, layer: 3, pos: 642
type: B, layer: 3, pos: 702
type: B, layer: 3, pos: 863
type: B, layer: 3, pos: 983
type: B, layer: 3, pos: 257
type: B, layer: 3, pos: 328
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 50
type: B, layer: 3, pos: 343
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 965
type: B, layer: 3, pos: 365
type: B, layer: 3, pos: 354
type: B, layer: 3, pos: 664
type: B, layer: 3, pos: 855
type: B, layer: 3, pos: 252
type: B, layer: 3, pos: 428
type: B, layer: 3, pos: 86
type: B, layer: 3, pos: 246
type: B, layer: 3, pos: 598
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 643
type: B, layer: 3, pos: 238
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 333
type: B, layer: 3, pos: 262
type: B, layer: 3, pos: 264
type: B, layer: 3, pos: 982
type: B, layer: 3, pos: 1012
type: B, layer: 3, pos: 124
type: B, layer: 3, pos: 648
type: B, layer: 3, pos: 641
type: B, layer: 3, pos: 857
type: B, layer: 3, pos: 985
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 77
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 372
type: B, layer: 3, pos: 666
type: B, layer: 3, pos: 322
type: B, layer: 3, pos: 84
type: B, layer: 3, pos: 81
type: B, layer: 3, pos: 873
type: B, layer: 3, pos: 665
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 324
type: B, layer: 3, pos: 109
type: B, layer: 3, pos: 1007
type: B, layer: 3, pos: 844
type: B, layer: 3, pos: 82
type: B, layer: 3, pos: 663
type: B, layer: 3, pos: 589
type: B, layer: 3, pos: 696
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 209
type: B, layer: 3, pos: 94
type: B, layer: 3, pos: 296
type: B, layer: 3, pos: 990
type: B, layer: 3, pos: 251
type: B, layer: 3, pos: 885
type: B, layer: 3, pos: 989
type: B, layer: 3, pos: 261
type: B, layer: 3, pos: 104
type: B, layer: 3, pos: 681
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 358
type: B, layer: 3, pos: 853
type: B, layer: 3, pos: 304
type: B, layer: 3, pos: 280
type: B, layer: 3, pos: 127
type: B, layer: 3, pos: 628
type: B, layer: 3, pos: 421
type: B, layer: 3, pos: 249
type: B, layer: 3, pos: 674
type: B, layer: 3, pos: 833
type: B, layer: 3, pos: 998
type: B, layer: 3, pos: 986
type: B, layer: 3, pos: 610
type: B, layer: 3, pos: 207
type: B, layer: 3, pos: 847
type: B, layer: 3, pos: 123
type: B, layer: 3, pos: 53
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 597
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 672
type: B, layer: 3, pos: 270
type: B, layer: 3, pos: 590
type: B, layer: 3, pos: 202
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 282
type: B, layer: 3, pos: 368
type: B, layer: 3, pos: 596
type: B, layer: 3, pos: 1001
type: B, layer: 3, pos: 364
type: B, layer: 3, pos: 321
type: B, layer: 3, pos: 126
type: B, layer: 3, pos: 834
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 362
type: B, layer: 3, pos: 970
type: B, layer: 3, pos: 275
type: B, layer: 3, pos: 1002
type: B, layer: 3, pos: 617
type: B, layer: 3, pos: 973
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 630
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 352
type: B, layer: 3, pos: 205
type: B, layer: 3, pos: 97
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 1006
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 606
type: B, layer: 3, pos: 52
type: B, layer: 3, pos: 112
type: B, layer: 3, pos: 276
type: B, layer: 3, pos: 845
type: B, layer: 3, pos: 602
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 119
type: B, layer: 3, pos: 588
type: B, layer: 3, pos: 215
type: B, layer: 3, pos: 320
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 217
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 593
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 871
type: B, layer: 3, pos: 301
type: B, layer: 3, pos: 979
type: B, layer: 3, pos: 594
type: B, layer: 3, pos: 629
type: B, layer: 3, pos: 1022
type: B, layer: 3, pos: 866
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1008
type: B, layer: 3, pos: 89
type: B, layer: 3, pos: 114
type: B, layer: 3, pos: 87
type: B, layer: 3, pos: 862
type: B, layer: 3, pos: 587
type: B, layer: 3, pos: 580
type: B, layer: 3, pos: 88
type: B, layer: 3, pos: 966
type: B, layer: 3, pos: 210
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 960
type: B, layer: 3, pos: 852
type: B, layer: 3, pos: 694
type: B, layer: 3, pos: 692
type: B, layer: 3, pos: 371
type: B, layer: 3, pos: 1016
type: B, layer: 3, pos: 247
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 591
type: B, layer: 3, pos: 599
type: B, layer: 3, pos: 103
type: B, layer: 3, pos: 288
type: B, layer: 3, pos: 351
type: B, layer: 3, pos: 341
type: B, layer: 3, pos: 631
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 267
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 367
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 256
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 266
type: B, layer: 3, pos: 271
type: B, layer: 3, pos: 413
type: B, layer: 3, pos: 653
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 586
type: B, layer: 3, pos: 585
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 682
type: B, layer: 3, pos: 105
type: B, layer: 3, pos: 601
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 623
type: B, layer: 3, pos: 110
type: B, layer: 3, pos: 968
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 701
type: B, layer: 3, pos: 691
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 1011
type: B, layer: 3, pos: 685
type: B, layer: 3, pos: 125
type: B, layer: 3, pos: 639
type: B, layer: 3, pos: 74
type: B, layer: 3, pos: 650
type: B, layer: 3, pos: 687
type: B, layer: 3, pos: 680
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 269
type: B, layer: 3, pos: 285
type: B, layer: 3, pos: 303
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 618
type: B, layer: 3, pos: 609
type: B, layer: 3, pos: 622
type: B, layer: 3, pos: 636
type: B, layer: 3, pos: 405
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 607
type: B, layer: 3, pos: 243
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 626

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 237

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1596706, upper bound: 14.5212909
time: 53.62 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1596706, upper bound: 14.6366808
time: 64.36 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -37.5422859, -0.1332417, -37.5021400, -0.3116074, -37.2306786, 37.3688965
1: -17.5822868, 10.5929489, -17.5598965, 10.4636431, -28.0459290, 28.1528454
2: -14.3703918, 10.4264069, -14.3152723, 10.0050354, -24.3754272, 24.7416801
3: -14.8746309, 14.3723402, -14.8641157, 14.0317268, -28.9063568, 29.2364559
4: -14.9897289, 15.2378902, -14.9070873, 14.6163855, -29.6061134, 30.1449776
5: -14.2031231, 15.5439949, -14.1793737, 15.1582985, -29.3614216, 29.7233696
6: -20.9407959, 10.0743837, -20.6717377, 10.0909739, -31.0317688, 30.7461205
7: -17.3661079, 16.7005653, -17.3153229, 16.5131264, -33.4990692, 33.6487579
8: -16.2041473, 19.5262566, -16.1155300, 19.0269928, -35.1620140, 35.6071358
9: -15.1855221, 13.6630640, -15.0155668, 13.5849304, -28.5796661, 28.5073910
10: -23.9479065, 17.1106548, -23.2345009, 16.9229584, -40.8708649, 40.3451538
11: -26.7066154, 10.1474676, -26.0716248, 10.1351795, -36.8417969, 36.2190933
12: -24.9656086, 12.0138035, -24.0833092, 11.9096336, -36.8752441, 36.0971146
13: -22.2376080, 18.4064331, -22.1628075, 18.3633919, -40.6009979, 40.5692406
14: -48.2557526, -0.4125519, -47.8115082, -0.4293175, -47.6635971, 47.2120819
15: -19.4056816, 10.4161396, -19.4335327, 10.2614918, -29.6671734, 29.8496723
16: -25.0652657, 12.9391718, -24.7682304, 12.9485826, -37.5682602, 37.2742653
17: -44.5161743, 12.3757820, -43.9163475, 12.3338861, -55.5541916, 54.9900970
18: -20.4081936, 12.4686184, -20.3548851, 12.4368877, -32.8450813, 32.8235016
19: -18.1308594, 4.2310138, -17.8561840, 4.2463665, -22.3772259, 22.0871983
20: -15.3537645, 8.4535999, -15.2367764, 8.4561396, -23.8099041, 23.6903763
21: -26.1856079, 3.7431257, -25.8214836, 3.7462509, -29.9318581, 29.5646095
22: -32.9171333, -0.8783064, -32.8105087, -0.9401407, -30.7398720, 30.5809250
23: -18.1430492, 8.8749933, -17.8793240, 8.8733139, -27.0163631, 26.7543182
24: -25.1902084, 7.3665643, -25.1358662, 7.2222195, -30.9542007, 31.0126381
25: -18.3671341, 10.8442183, -18.2451649, 10.7993841, -29.1665192, 29.0893822
26: -23.9360580, 14.8913527, -23.6392403, 14.8616495, -38.7977066, 38.5305939
27: -26.2167053, 6.7071571, -26.1408539, 6.6243591, -31.8779221, 31.8416691
28: -17.4332314, 10.5880232, -17.2926483, 10.6014252, -27.8421822, 27.6743717
29: -40.4299240, -5.3118896, -40.1126556, -5.3118343, -34.2008514, 33.8216705
30: -21.0010376, 12.2450562, -20.8397484, 12.2809143, -33.2819519, 33.0848045
31: -23.8263111, 6.9879670, -23.6363869, 6.9822598, -30.8085709, 30.6243534
32: -27.7971687, 4.2567043, -27.5277443, 4.2428284, -31.1607819, 30.8647003
33: -30.5144920, 14.8503866, -30.4405041, 14.5279636, -44.0750008, 44.4342804
34: -25.9840622, 9.9822025, -25.9302101, 9.8752222, -35.8592834, 35.9124146
35: -27.7137451, 11.0931091, -27.6267319, 10.8676920, -38.1192894, 38.3921089
36: -27.2145844, 10.8664970, -27.1250801, 10.8639507, -37.6394234, 37.5469055
37: -37.3122177, 9.5912085, -37.1635017, 9.5986004, -45.6494598, 45.4572105
38: -29.7006321, 14.0187435, -29.6173820, 13.9508553, -43.6514893, 43.6361237
39: -38.4470596, 11.7421894, -38.3146477, 11.5719547, -49.3715363, 49.3997650
40: -30.4065704, 9.8245964, -30.3690205, 9.7589760, -38.5075150, 38.5317001
41: -22.4882240, 9.3549919, -22.2831974, 9.3977680, -31.8859921, 31.6381893
42: -16.7503853, 7.3910441, -16.1804085, 7.2984324, -23.7648602, 23.2353210

Time for backsubstitution: 0.95 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 292
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 356
type: B, layer: 3, pos: 229
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 868
type: B, layer: 3, pos: 355
type: B, layer: 3, pos: 357
type: B, layer: 3, pos: 363
type: B, layer: 3, pos: 348
type: B, layer: 3, pos: 284
type: B, layer: 3, pos: 869
type: B, layer: 3, pos: 997
type: B, layer: 3, pos: 887
type: B, layer: 3, pos: 377
type: B, layer: 3, pos: 353
type: B, layer: 3, pos: 369
type: B, layer: 3, pos: 375
type: B, layer: 3, pos: 875
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 293
type: B, layer: 3, pos: 988
type: B, layer: 3, pos: 999
type: B, layer: 3, pos: 881
type: B, layer: 3, pos: 291
type: B, layer: 3, pos: 378
type: B, layer: 3, pos: 991
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 283
type: B, layer: 3, pos: 305
type: B, layer: 3, pos: 996
type: B, layer: 3, pos: 289
type: B, layer: 3, pos: 383
type: B, layer: 3, pos: 993
type: B, layer: 3, pos: 380
type: B, layer: 3, pos: 1009
type: B, layer: 3, pos: 893
type: B, layer: 3, pos: 877
type: B, layer: 3, pos: 331
type: B, layer: 3, pos: 339
type: B, layer: 3, pos: 361
type: B, layer: 3, pos: 849
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 338
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 1015
type: B, layer: 3, pos: 972
type: B, layer: 3, pos: 850
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 231
type: B, layer: 3, pos: 889
type: B, layer: 3, pos: 865
type: B, layer: 3, pos: 843
type: B, layer: 3, pos: 684
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 300
type: B, layer: 3, pos: 859
type: B, layer: 3, pos: 895
type: B, layer: 3, pos: 882
type: B, layer: 3, pos: 347
type: B, layer: 3, pos: 689
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 841
type: B, layer: 3, pos: 1023
type: B, layer: 3, pos: 379
type: B, layer: 3, pos: 644
type: B, layer: 3, pos: 382
type: B, layer: 3, pos: 860
type: B, layer: 3, pos: 695
type: B, layer: 3, pos: 223
type: B, layer: 3, pos: 239
type: B, layer: 3, pos: 346
type: B, layer: 3, pos: 329
type: B, layer: 3, pos: 306
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 1003
type: B, layer: 3, pos: 265
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 313
type: B, layer: 3, pos: 334
type: B, layer: 3, pos: 273
type: B, layer: 3, pos: 314
type: B, layer: 3, pos: 85
type: B, layer: 3, pos: 978
type: B, layer: 3, pos: 874
type: B, layer: 3, pos: 1005
type: B, layer: 3, pos: 58
type: B, layer: 3, pos: 1021
type: B, layer: 3, pos: 846
type: B, layer: 3, pos: 69
type: B, layer: 3, pos: 884
type: B, layer: 3, pos: 647
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 699
type: B, layer: 3, pos: 1017
type: B, layer: 3, pos: 274
type: B, layer: 3, pos: 977
type: B, layer: 3, pos: 299
type: B, layer: 3, pos: 894
type: B, layer: 3, pos: 974
type: B, layer: 3, pos: 370
type: B, layer: 3, pos: 995
type: B, layer: 3, pos: 851
type: B, layer: 3, pos: 646
type: B, layer: 3, pos: 698
type: B, layer: 3, pos: 876
type: B, layer: 3, pos: 667
type: B, layer: 3, pos: 260
type: B, layer: 3, pos: 381
type: B, layer: 3, pos: 235
type: B, layer: 3, pos: 1019
type: B, layer: 3, pos: 673
type: B, layer: 3, pos: 316
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 867
type: B, layer: 3, pos: 419
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 980
type: B, layer: 3, pos: 967
type: B, layer: 3, pos: 315
type: B, layer: 3, pos: 319
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 258
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 376
type: B, layer: 3, pos: 259
type: B, layer: 3, pos: 883
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 700
type: B, layer: 3, pos: 836
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 1020
type: B, layer: 3, pos: 842
type: B, layer: 3, pos: 61
type: B, layer: 3, pos: 1018
type: B, layer: 3, pos: 1014
type: B, layer: 3, pos: 688
type: B, layer: 3, pos: 336
type: B, layer: 3, pos: 56
type: B, layer: 3, pos: 1010
type: B, layer: 3, pos: 272
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 51
type: B, layer: 3, pos: 656
type: B, layer: 3, pos: 345
type: B, layer: 3, pos: 340
type: B, layer: 3, pos: 975
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 657
type: B, layer: 3, pos: 201
type: B, layer: 3, pos: 360
type: B, layer: 3, pos: 645
type: B, layer: 3, pos: 690
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 649
type: B, layer: 3, pos: 220
type: B, layer: 3, pos: 683
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 335
type: B, layer: 3, pos: 62
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 703
type: B, layer: 3, pos: 102
type: B, layer: 3, pos: 870
type: B, layer: 3, pos: 344
type: B, layer: 3, pos: 68
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 337
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 111
type: B, layer: 3, pos: 404
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 651
type: B, layer: 3, pos: 1013
type: B, layer: 3, pos: 349
type: B, layer: 3, pos: 858
type: B, layer: 3, pos: 981
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 242
type: B, layer: 3, pos: 861
type: B, layer: 3, pos: 1004
type: B, layer: 3, pos: 987
type: B, layer: 3, pos: 325
type: B, layer: 3, pos: 279
type: B, layer: 3, pos: 281
type: B, layer: 3, pos: 113
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 78
type: B, layer: 3, pos: 658
type: B, layer: 3, pos: 57
type: B, layer: 3, pos: 54
type: B, layer: 3, pos: 297
type: B, layer: 3, pos: 203
type: B, layer: 3, pos: 835
type: B, layer: 3, pos: 971
type: B, layer: 3, pos: 420
type: B, layer: 3, pos: 63
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 55
type: B, layer: 3, pos: 879
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 595
type: B, layer: 3, pos: 263
type: B, layer: 3, pos: 642
type: B, layer: 3, pos: 702
type: B, layer: 3, pos: 863
type: B, layer: 3, pos: 969
type: B, layer: 3, pos: 318
type: B, layer: 3, pos: 983
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 328
type: B, layer: 3, pos: 257
type: B, layer: 3, pos: 50
type: B, layer: 3, pos: 965
type: B, layer: 3, pos: 343
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 365
type: B, layer: 3, pos: 855
type: B, layer: 3, pos: 86
type: B, layer: 3, pos: 428
type: B, layer: 3, pos: 664
type: B, layer: 3, pos: 246
type: B, layer: 3, pos: 354
type: B, layer: 3, pos: 598
type: B, layer: 3, pos: 252
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 643
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 238
type: B, layer: 3, pos: 333
type: B, layer: 3, pos: 1012
type: B, layer: 3, pos: 264
type: B, layer: 3, pos: 124
type: B, layer: 3, pos: 982
type: B, layer: 3, pos: 262
type: B, layer: 3, pos: 648
type: B, layer: 3, pos: 641
type: B, layer: 3, pos: 985
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 77
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 857
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 372
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 666
type: B, layer: 3, pos: 322
type: B, layer: 3, pos: 84
type: B, layer: 3, pos: 81
type: B, layer: 3, pos: 873
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 665
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 324
type: B, layer: 3, pos: 109
type: B, layer: 3, pos: 844
type: B, layer: 3, pos: 82
type: B, layer: 3, pos: 1007
type: B, layer: 3, pos: 589
type: B, layer: 3, pos: 663
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 696
type: B, layer: 3, pos: 209
type: B, layer: 3, pos: 94
type: B, layer: 3, pos: 296
type: B, layer: 3, pos: 989
type: B, layer: 3, pos: 251
type: B, layer: 3, pos: 885
type: B, layer: 3, pos: 990
type: B, layer: 3, pos: 104
type: B, layer: 3, pos: 261
type: B, layer: 3, pos: 681
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 127
type: B, layer: 3, pos: 358
type: B, layer: 3, pos: 304
type: B, layer: 3, pos: 421
type: B, layer: 3, pos: 280
type: B, layer: 3, pos: 628
type: B, layer: 3, pos: 853
type: B, layer: 3, pos: 249
type: B, layer: 3, pos: 833
type: B, layer: 3, pos: 674
type: B, layer: 3, pos: 207
type: B, layer: 3, pos: 610
type: B, layer: 3, pos: 986
type: B, layer: 3, pos: 123
type: B, layer: 3, pos: 53
type: B, layer: 3, pos: 847
type: B, layer: 3, pos: 998
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 597
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 672
type: B, layer: 3, pos: 270
type: B, layer: 3, pos: 590
type: B, layer: 3, pos: 202
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 368
type: B, layer: 3, pos: 282
type: B, layer: 3, pos: 596
type: B, layer: 3, pos: 1001
type: B, layer: 3, pos: 321
type: B, layer: 3, pos: 364
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 834
type: B, layer: 3, pos: 126
type: B, layer: 3, pos: 970
type: B, layer: 3, pos: 362
type: B, layer: 3, pos: 275
type: B, layer: 3, pos: 973
type: B, layer: 3, pos: 617
type: B, layer: 3, pos: 1002
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 630
type: B, layer: 3, pos: 352
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 205
type: B, layer: 3, pos: 97
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1006
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 276
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 52
type: B, layer: 3, pos: 606
type: B, layer: 3, pos: 112
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 845
type: B, layer: 3, pos: 602
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 588
type: B, layer: 3, pos: 320
type: B, layer: 3, pos: 119
type: B, layer: 3, pos: 215
type: B, layer: 3, pos: 217
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 593
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 871
type: B, layer: 3, pos: 979
type: B, layer: 3, pos: 594
type: B, layer: 3, pos: 629
type: B, layer: 3, pos: 301
type: B, layer: 3, pos: 1022
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 866
type: B, layer: 3, pos: 1008
type: B, layer: 3, pos: 89
type: B, layer: 3, pos: 114
type: B, layer: 3, pos: 87
type: B, layer: 3, pos: 862
type: B, layer: 3, pos: 587
type: B, layer: 3, pos: 580
type: B, layer: 3, pos: 88
type: B, layer: 3, pos: 966
type: B, layer: 3, pos: 210
type: B, layer: 3, pos: 852
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 960
type: B, layer: 3, pos: 694
type: B, layer: 3, pos: 692
type: B, layer: 3, pos: 371
type: B, layer: 3, pos: 1016
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 591
type: B, layer: 3, pos: 247
type: B, layer: 3, pos: 599
type: B, layer: 3, pos: 103
type: B, layer: 3, pos: 351
type: B, layer: 3, pos: 288
type: B, layer: 3, pos: 631
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 341
type: B, layer: 3, pos: 267
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 367
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 256
type: B, layer: 3, pos: 271
type: B, layer: 3, pos: 266
type: B, layer: 3, pos: 413
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 653
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 586
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 585
type: B, layer: 3, pos: 682
type: B, layer: 3, pos: 105
type: B, layer: 3, pos: 601
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 623
type: B, layer: 3, pos: 110
type: B, layer: 3, pos: 968
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 701
type: B, layer: 3, pos: 691
type: B, layer: 3, pos: 1011
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 685
type: B, layer: 3, pos: 125
type: B, layer: 3, pos: 639
type: B, layer: 3, pos: 650
type: B, layer: 3, pos: 74
type: B, layer: 3, pos: 687
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 680
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 269
type: B, layer: 3, pos: 285
type: B, layer: 3, pos: 303
type: B, layer: 3, pos: 618
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 609
type: B, layer: 3, pos: 622
type: B, layer: 3, pos: 636
type: B, layer: 3, pos: 405
type: B, layer: 3, pos: 607
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 243
type: B, layer: 3, pos: 626

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 292

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 228

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1051552, upper bound: 14.3698167
time: 54.84 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1082246, upper bound: 14.4897330
time: 57.03 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -37.5427055, -0.0906982, -37.4917564, -0.3116684, -37.2310371, 37.4010582
1: -17.5720005, 10.6087351, -17.5460815, 10.4621429, -28.0341434, 28.1548157
2: -14.3700342, 10.4881611, -14.3014545, 10.0038633, -24.3738976, 24.7896156
3: -14.8911867, 14.4251308, -14.8573179, 14.0304518, -28.9216385, 29.2824478
4: -15.1093197, 15.4872484, -14.9111080, 14.6155128, -29.7248325, 30.3983574
5: -14.2014380, 15.6136246, -14.1674128, 15.1574726, -29.3589096, 29.7810364
6: -21.2161064, 10.2000160, -20.6707268, 10.1042385, -31.3203449, 30.8707428
7: -17.4006729, 16.7075787, -17.3134403, 16.5100460, -33.5344467, 33.6518593
8: -16.2731171, 19.6652622, -16.1048603, 19.0240669, -35.2322388, 35.7483063
9: -15.2072258, 13.6721973, -15.0125208, 13.5701351, -28.5815277, 28.5092735
10: -24.0409184, 17.1565819, -23.2312679, 16.9169693, -40.9578857, 40.3878479
11: -27.0157833, 10.2605286, -26.0716839, 10.1306372, -37.1464195, 36.3322144
12: -25.1449738, 12.0901413, -24.0820389, 11.9070435, -37.0520172, 36.1721802
13: -22.2539597, 18.4485664, -22.1600361, 18.3599415, -40.6138992, 40.6086044
14: -48.2337952, -0.3990955, -47.7917290, -0.4310017, -47.6379089, 47.1990623
15: -19.5540333, 10.6849060, -19.4430962, 10.2612820, -29.8153152, 30.1280022
16: -25.2675171, 12.9967499, -24.7648010, 12.9328938, -37.7354126, 37.3168259
17: -44.5991554, 12.4159317, -43.9138718, 12.3255367, -55.6284485, 55.0346222
18: -20.4109344, 12.5100594, -20.3416386, 12.4350691, -32.8460045, 32.8516998
19: -18.2330513, 4.2552900, -17.8544941, 4.2428837, -22.4759350, 22.1097832
20: -15.3888521, 8.4720011, -15.2346315, 8.4556293, -23.8444824, 23.7066326
21: -26.2934799, 3.7663136, -25.8201790, 3.7425213, -30.0360012, 29.5864925
22: -32.9689102, -0.7977896, -32.8040581, -0.9403000, -30.7713242, 30.6294975
23: -18.2286053, 8.9036865, -17.8772697, 8.8666210, -27.0952263, 26.7809563
24: -25.2292118, 7.4168949, -25.1340160, 7.2212744, -30.9905815, 31.0481567
25: -18.3712845, 10.8802567, -18.2303162, 10.7980719, -29.1693573, 29.1105728
26: -23.9454803, 14.9143982, -23.6227665, 14.8595667, -38.8050461, 38.5371628
27: -26.2534981, 6.7250633, -26.1338291, 6.6227665, -31.9201775, 31.8514137
28: -17.5157433, 10.6136589, -17.2917633, 10.5983391, -27.9313927, 27.7074165
29: -40.4749374, -5.2782135, -40.1103783, -5.3152466, -34.2755280, 33.8487053
30: -21.0932751, 12.2920570, -20.8394470, 12.2772226, -33.3704987, 33.1315041
31: -23.8895226, 7.0021658, -23.6338310, 6.9788847, -30.8684082, 30.6359978
32: -27.9563293, 4.3184195, -27.5268917, 4.2384148, -31.3510818, 30.9439697
33: -30.5479698, 14.8932877, -30.4337215, 14.5277290, -44.1038666, 44.4833488
34: -25.9991245, 9.9769096, -25.9281044, 9.8625832, -35.8617096, 35.9050140
35: -27.7394524, 11.1164227, -27.6231155, 10.8658981, -38.1388855, 38.4251137
36: -27.2481251, 10.8604851, -27.1240368, 10.8537884, -37.6726570, 37.5466537
37: -37.3441582, 9.5457506, -37.1599197, 9.5682592, -45.6659851, 45.4317665
38: -29.7275925, 14.0488148, -29.6083107, 13.9484339, -43.6760254, 43.6571274
39: -38.5085945, 11.7644615, -38.3095245, 11.5717716, -49.4339600, 49.4163780
40: -30.4417229, 9.7196379, -30.3662949, 9.7134285, -38.5001984, 38.4315071
41: -22.6734943, 9.4087410, -22.2812080, 9.3900127, -32.0635071, 31.6899490
42: -17.0084686, 7.5196338, -16.1792831, 7.3118563, -24.0534592, 23.3603191

Time for backsubstitution: 0.95 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 292
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 356
type: B, layer: 3, pos: 229
type: B, layer: 3, pos: 868
type: B, layer: 3, pos: 355
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 357
type: B, layer: 3, pos: 363
type: B, layer: 3, pos: 348
type: B, layer: 3, pos: 284
type: B, layer: 3, pos: 869
type: B, layer: 3, pos: 997
type: B, layer: 3, pos: 887
type: B, layer: 3, pos: 377
type: B, layer: 3, pos: 353
type: B, layer: 3, pos: 369
type: B, layer: 3, pos: 375
type: B, layer: 3, pos: 875
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 293
type: B, layer: 3, pos: 988
type: B, layer: 3, pos: 999
type: B, layer: 3, pos: 881
type: B, layer: 3, pos: 291
type: B, layer: 3, pos: 378
type: B, layer: 3, pos: 991
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 283
type: B, layer: 3, pos: 305
type: B, layer: 3, pos: 996
type: B, layer: 3, pos: 289
type: B, layer: 3, pos: 383
type: B, layer: 3, pos: 993
type: B, layer: 3, pos: 380
type: B, layer: 3, pos: 1009
type: B, layer: 3, pos: 893
type: B, layer: 3, pos: 877
type: B, layer: 3, pos: 331
type: B, layer: 3, pos: 339
type: B, layer: 3, pos: 361
type: B, layer: 3, pos: 849
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 338
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 1015
type: B, layer: 3, pos: 972
type: B, layer: 3, pos: 850
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 231
type: B, layer: 3, pos: 889
type: B, layer: 3, pos: 865
type: B, layer: 3, pos: 843
type: B, layer: 3, pos: 684
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 300
type: B, layer: 3, pos: 859
type: B, layer: 3, pos: 895
type: B, layer: 3, pos: 882
type: B, layer: 3, pos: 347
type: B, layer: 3, pos: 689
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 841
type: B, layer: 3, pos: 1023
type: B, layer: 3, pos: 379
type: B, layer: 3, pos: 644
type: B, layer: 3, pos: 382
type: B, layer: 3, pos: 860
type: B, layer: 3, pos: 695
type: B, layer: 3, pos: 223
type: B, layer: 3, pos: 239
type: B, layer: 3, pos: 346
type: B, layer: 3, pos: 329
type: B, layer: 3, pos: 306
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 1003
type: B, layer: 3, pos: 265
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 313
type: B, layer: 3, pos: 334
type: B, layer: 3, pos: 273
type: B, layer: 3, pos: 314
type: B, layer: 3, pos: 85
type: B, layer: 3, pos: 978
type: B, layer: 3, pos: 874
type: B, layer: 3, pos: 1005
type: B, layer: 3, pos: 58
type: B, layer: 3, pos: 1021
type: B, layer: 3, pos: 846
type: B, layer: 3, pos: 69
type: B, layer: 3, pos: 884
type: B, layer: 3, pos: 647
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 699
type: B, layer: 3, pos: 1017
type: B, layer: 3, pos: 274
type: B, layer: 3, pos: 977
type: B, layer: 3, pos: 299
type: B, layer: 3, pos: 894
type: B, layer: 3, pos: 974
type: B, layer: 3, pos: 995
type: B, layer: 3, pos: 370
type: B, layer: 3, pos: 851
type: B, layer: 3, pos: 646
type: B, layer: 3, pos: 698
type: B, layer: 3, pos: 876
type: B, layer: 3, pos: 667
type: B, layer: 3, pos: 260
type: B, layer: 3, pos: 381
type: B, layer: 3, pos: 235
type: B, layer: 3, pos: 1019
type: B, layer: 3, pos: 673
type: B, layer: 3, pos: 316
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 867
type: B, layer: 3, pos: 419
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 980
type: B, layer: 3, pos: 967
type: B, layer: 3, pos: 315
type: B, layer: 3, pos: 319
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 258
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 376
type: B, layer: 3, pos: 259
type: B, layer: 3, pos: 883
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 700
type: B, layer: 3, pos: 836
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 1020
type: B, layer: 3, pos: 842
type: B, layer: 3, pos: 61
type: B, layer: 3, pos: 1018
type: B, layer: 3, pos: 1014
type: B, layer: 3, pos: 688
type: B, layer: 3, pos: 336
type: B, layer: 3, pos: 56
type: B, layer: 3, pos: 1010
type: B, layer: 3, pos: 272
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 51
type: B, layer: 3, pos: 656
type: B, layer: 3, pos: 345
type: B, layer: 3, pos: 340
type: B, layer: 3, pos: 975
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 657
type: B, layer: 3, pos: 201
type: B, layer: 3, pos: 360
type: B, layer: 3, pos: 645
type: B, layer: 3, pos: 690
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 649
type: B, layer: 3, pos: 220
type: B, layer: 3, pos: 683
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 335
type: B, layer: 3, pos: 62
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 703
type: B, layer: 3, pos: 102
type: B, layer: 3, pos: 870
type: B, layer: 3, pos: 344
type: B, layer: 3, pos: 68
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 337
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 111
type: B, layer: 3, pos: 404
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 651
type: B, layer: 3, pos: 1013
type: B, layer: 3, pos: 349
type: B, layer: 3, pos: 858
type: B, layer: 3, pos: 981
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 242
type: B, layer: 3, pos: 861
type: B, layer: 3, pos: 1004
type: B, layer: 3, pos: 987
type: B, layer: 3, pos: 325
type: B, layer: 3, pos: 279
type: B, layer: 3, pos: 113
type: B, layer: 3, pos: 281
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 78
type: B, layer: 3, pos: 658
type: B, layer: 3, pos: 57
type: B, layer: 3, pos: 54
type: B, layer: 3, pos: 297
type: B, layer: 3, pos: 203
type: B, layer: 3, pos: 835
type: B, layer: 3, pos: 971
type: B, layer: 3, pos: 420
type: B, layer: 3, pos: 63
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 55
type: B, layer: 3, pos: 879
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 595
type: B, layer: 3, pos: 263
type: B, layer: 3, pos: 642
type: B, layer: 3, pos: 702
type: B, layer: 3, pos: 863
type: B, layer: 3, pos: 969
type: B, layer: 3, pos: 318
type: B, layer: 3, pos: 983
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 328
type: B, layer: 3, pos: 257
type: B, layer: 3, pos: 50
type: B, layer: 3, pos: 965
type: B, layer: 3, pos: 343
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 365
type: B, layer: 3, pos: 855
type: B, layer: 3, pos: 86
type: B, layer: 3, pos: 428
type: B, layer: 3, pos: 664
type: B, layer: 3, pos: 246
type: B, layer: 3, pos: 354
type: B, layer: 3, pos: 598
type: B, layer: 3, pos: 252
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 643
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 238
type: B, layer: 3, pos: 333
type: B, layer: 3, pos: 1012
type: B, layer: 3, pos: 264
type: B, layer: 3, pos: 124
type: B, layer: 3, pos: 982
type: B, layer: 3, pos: 262
type: B, layer: 3, pos: 648
type: B, layer: 3, pos: 641
type: B, layer: 3, pos: 985
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 77
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 857
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 372
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 666
type: B, layer: 3, pos: 322
type: B, layer: 3, pos: 84
type: B, layer: 3, pos: 81
type: B, layer: 3, pos: 873
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 665
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 324
type: B, layer: 3, pos: 109
type: B, layer: 3, pos: 844
type: B, layer: 3, pos: 82
type: B, layer: 3, pos: 1007
type: B, layer: 3, pos: 589
type: B, layer: 3, pos: 663
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 696
type: B, layer: 3, pos: 209
type: B, layer: 3, pos: 94
type: B, layer: 3, pos: 296
type: B, layer: 3, pos: 989
type: B, layer: 3, pos: 885
type: B, layer: 3, pos: 251
type: B, layer: 3, pos: 990
type: B, layer: 3, pos: 104
type: B, layer: 3, pos: 261
type: B, layer: 3, pos: 681
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 127
type: B, layer: 3, pos: 358
type: B, layer: 3, pos: 304
type: B, layer: 3, pos: 421
type: B, layer: 3, pos: 280
type: B, layer: 3, pos: 628
type: B, layer: 3, pos: 853
type: B, layer: 3, pos: 249
type: B, layer: 3, pos: 833
type: B, layer: 3, pos: 674
type: B, layer: 3, pos: 207
type: B, layer: 3, pos: 610
type: B, layer: 3, pos: 986
type: B, layer: 3, pos: 123
type: B, layer: 3, pos: 53
type: B, layer: 3, pos: 847
type: B, layer: 3, pos: 998
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 597
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 672
type: B, layer: 3, pos: 270
type: B, layer: 3, pos: 590
type: B, layer: 3, pos: 202
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 368
type: B, layer: 3, pos: 282
type: B, layer: 3, pos: 596
type: B, layer: 3, pos: 1001
type: B, layer: 3, pos: 321
type: B, layer: 3, pos: 364
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 834
type: B, layer: 3, pos: 126
type: B, layer: 3, pos: 970
type: B, layer: 3, pos: 362
type: B, layer: 3, pos: 275
type: B, layer: 3, pos: 973
type: B, layer: 3, pos: 617
type: B, layer: 3, pos: 1002
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 630
type: B, layer: 3, pos: 352
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 205
type: B, layer: 3, pos: 97
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 1006
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 276
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 52
type: B, layer: 3, pos: 606
type: B, layer: 3, pos: 112
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 845
type: B, layer: 3, pos: 602
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 588
type: B, layer: 3, pos: 320
type: B, layer: 3, pos: 119
type: B, layer: 3, pos: 215
type: B, layer: 3, pos: 217
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 593
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 871
type: B, layer: 3, pos: 979
type: B, layer: 3, pos: 594
type: B, layer: 3, pos: 629
type: B, layer: 3, pos: 301
type: B, layer: 3, pos: 1022
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 866
type: B, layer: 3, pos: 1008
type: B, layer: 3, pos: 89
type: B, layer: 3, pos: 114
type: B, layer: 3, pos: 87
type: B, layer: 3, pos: 862
type: B, layer: 3, pos: 587
type: B, layer: 3, pos: 580
type: B, layer: 3, pos: 88
type: B, layer: 3, pos: 966
type: B, layer: 3, pos: 210
type: B, layer: 3, pos: 852
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 960
type: B, layer: 3, pos: 694
type: B, layer: 3, pos: 692
type: B, layer: 3, pos: 371
type: B, layer: 3, pos: 1016
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 591
type: B, layer: 3, pos: 247
type: B, layer: 3, pos: 599
type: B, layer: 3, pos: 103
type: B, layer: 3, pos: 351
type: B, layer: 3, pos: 288
type: B, layer: 3, pos: 631
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 341
type: B, layer: 3, pos: 267
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 367
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 256
type: B, layer: 3, pos: 271
type: B, layer: 3, pos: 266
type: B, layer: 3, pos: 413
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 653
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 586
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 585
type: B, layer: 3, pos: 682
type: B, layer: 3, pos: 105
type: B, layer: 3, pos: 601
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 623
type: B, layer: 3, pos: 110
type: B, layer: 3, pos: 968
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 701
type: B, layer: 3, pos: 691
type: B, layer: 3, pos: 1011
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 685
type: B, layer: 3, pos: 125
type: B, layer: 3, pos: 639
type: B, layer: 3, pos: 650
type: B, layer: 3, pos: 74
type: B, layer: 3, pos: 687
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 680
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 269
type: B, layer: 3, pos: 285
type: B, layer: 3, pos: 303
type: B, layer: 3, pos: 618
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 609
type: B, layer: 3, pos: 622
type: B, layer: 3, pos: 636
type: B, layer: 3, pos: 405
type: B, layer: 3, pos: 607
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 243
type: B, layer: 3, pos: 626

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 292

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 228

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1051552, upper bound: 14.3698167
time: 64.92 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1082246, upper bound: 14.4897330
time: 59.25 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -37.5515099, -0.1297913, -37.5286217, -0.2904243, -37.2610855, 37.3988304
1: -17.5930252, 10.5972595, -17.5865631, 10.4757013, -28.0687256, 28.1838226
2: -14.3985147, 10.4302044, -14.3982763, 10.0828037, -24.4813194, 24.8284798
3: -14.8831234, 14.3788500, -14.8902702, 14.0680561, -28.9511795, 29.2691193
4: -15.0426149, 15.2417669, -15.0717010, 14.7487774, -29.7913933, 30.3134689
5: -14.2107840, 15.5477066, -14.2050982, 15.1872234, -29.3980064, 29.7528038
6: -20.9488411, 10.1305199, -20.8208351, 10.2527885, -31.2016296, 30.9513550
7: -17.3736782, 16.7026978, -17.3445740, 16.5226288, -33.5177002, 33.6799011
8: -16.2542896, 19.5331841, -16.2692585, 19.1600780, -35.3868027, 35.7694397
9: -15.1968880, 13.7100992, -15.1511078, 13.7266703, -28.6936646, 28.6780434
10: -23.9652748, 17.2103844, -23.5340767, 17.2104683, -41.1757431, 40.7444611
11: -26.7171974, 10.1956921, -26.2098351, 10.2802048, -36.9974022, 36.4055252
12: -24.9698601, 12.0637474, -24.2252979, 12.0607090, -37.0305710, 36.2890472
13: -22.2418861, 18.4193573, -22.1895924, 18.4107342, -40.6526184, 40.6089478
14: -48.2583885, -0.4053326, -47.8278770, -0.4326591, -47.6577148, 47.2374153
15: -19.4388847, 10.4259396, -19.5390644, 10.3163471, -29.7552319, 29.9650040
16: -25.0847282, 13.0076799, -24.9596367, 13.1481190, -37.7431030, 37.5259247
17: -44.5205917, 12.3927860, -43.9492683, 12.3905830, -55.6152649, 55.0397491
18: -20.4174557, 12.4820271, -20.3954391, 12.4746990, -32.8921547, 32.8774643
19: -18.1392250, 4.2384200, -17.9057884, 4.2672720, -22.4064980, 22.1442089
20: -15.3594189, 8.4578123, -15.2649031, 8.4661446, -23.8255634, 23.7227154
21: -26.1935444, 3.7499769, -25.8775864, 3.7655427, -29.9590874, 29.6275635
22: -32.9421806, -0.8695583, -32.8919792, -0.8881035, -30.7696953, 30.6554375
23: -18.1477928, 8.8873339, -17.9154949, 8.9132919, -27.0610847, 26.8028297
24: -25.2256813, 7.3684855, -25.2413864, 7.3379884, -31.0916977, 31.0377407
25: -18.3828049, 10.8480492, -18.3029327, 10.8175125, -29.2003174, 29.1509819
26: -23.9395523, 14.9042416, -23.6638985, 14.8878622, -38.8274155, 38.5681381
27: -26.2440987, 6.7112656, -26.2317448, 6.6833134, -31.9699631, 31.9037056
28: -17.4372101, 10.5912838, -17.3101883, 10.6206675, -27.8617821, 27.6999836
29: -40.4355965, -5.3068657, -40.1380043, -5.2975616, -34.2047882, 33.9131699
30: -21.0076199, 12.2502565, -20.8650322, 12.2999325, -33.3075523, 33.1152878
31: -23.8376770, 6.9905443, -23.6964283, 6.9905014, -30.8281784, 30.6869736
32: -27.8021851, 4.2937660, -27.6291237, 4.3512030, -31.2439041, 31.0122147
33: -30.5345039, 14.8568478, -30.5011292, 14.5875397, -44.1732712, 44.4973335
34: -25.9904747, 9.9934502, -25.9524059, 9.9162064, -35.9066811, 35.9458542
35: -27.7429352, 11.1014957, -27.7110748, 10.9598703, -38.2374535, 38.4721107
36: -27.2226238, 10.8725328, -27.1544495, 10.8895330, -37.6714668, 37.5797577
37: -37.3227158, 9.5956326, -37.2034950, 9.6188211, -45.6774902, 45.4989052
38: -29.7147808, 14.0284119, -29.6646271, 13.9880772, -43.7028580, 43.6930389
39: -38.4751396, 11.7446632, -38.4105606, 11.6147518, -49.4386139, 49.4844513
40: -30.4180832, 9.8168869, -30.3851852, 9.7363415, -38.5046844, 38.5419083
41: -22.4971409, 9.3851185, -22.3799019, 9.4796524, -31.9579735, 31.7650204
42: -16.7556324, 7.4678426, -16.3857632, 7.5274267, -23.8683701, 23.5170364

Time for backsubstitution: 0.95 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 292
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 356
type: B, layer: 3, pos: 229
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 868
type: B, layer: 3, pos: 355
type: B, layer: 3, pos: 357
type: B, layer: 3, pos: 363
type: B, layer: 3, pos: 348
type: B, layer: 3, pos: 284
type: B, layer: 3, pos: 869
type: B, layer: 3, pos: 997
type: B, layer: 3, pos: 887
type: B, layer: 3, pos: 377
type: B, layer: 3, pos: 353
type: B, layer: 3, pos: 369
type: B, layer: 3, pos: 375
type: B, layer: 3, pos: 875
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 293
type: B, layer: 3, pos: 988
type: B, layer: 3, pos: 999
type: B, layer: 3, pos: 881
type: B, layer: 3, pos: 291
type: B, layer: 3, pos: 378
type: B, layer: 3, pos: 991
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 305
type: B, layer: 3, pos: 283
type: B, layer: 3, pos: 996
type: B, layer: 3, pos: 289
type: B, layer: 3, pos: 383
type: B, layer: 3, pos: 993
type: B, layer: 3, pos: 380
type: B, layer: 3, pos: 1009
type: B, layer: 3, pos: 893
type: B, layer: 3, pos: 877
type: B, layer: 3, pos: 331
type: B, layer: 3, pos: 361
type: B, layer: 3, pos: 339
type: B, layer: 3, pos: 849
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 338
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 1015
type: B, layer: 3, pos: 972
type: B, layer: 3, pos: 850
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 231
type: B, layer: 3, pos: 889
type: B, layer: 3, pos: 865
type: B, layer: 3, pos: 843
type: B, layer: 3, pos: 684
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 300
type: B, layer: 3, pos: 859
type: B, layer: 3, pos: 895
type: B, layer: 3, pos: 882
type: B, layer: 3, pos: 347
type: B, layer: 3, pos: 689
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 1023
type: B, layer: 3, pos: 841
type: B, layer: 3, pos: 379
type: B, layer: 3, pos: 382
type: B, layer: 3, pos: 644
type: B, layer: 3, pos: 860
type: B, layer: 3, pos: 695
type: B, layer: 3, pos: 223
type: B, layer: 3, pos: 239
type: B, layer: 3, pos: 346
type: B, layer: 3, pos: 329
type: B, layer: 3, pos: 306
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 1003
type: B, layer: 3, pos: 265
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 313
type: B, layer: 3, pos: 334
type: B, layer: 3, pos: 273
type: B, layer: 3, pos: 314
type: B, layer: 3, pos: 85
type: B, layer: 3, pos: 978
type: B, layer: 3, pos: 874
type: B, layer: 3, pos: 1005
type: B, layer: 3, pos: 58
type: B, layer: 3, pos: 1021
type: B, layer: 3, pos: 846
type: B, layer: 3, pos: 69
type: B, layer: 3, pos: 884
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 647
type: B, layer: 3, pos: 1017
type: B, layer: 3, pos: 699
type: B, layer: 3, pos: 274
type: B, layer: 3, pos: 977
type: B, layer: 3, pos: 299
type: B, layer: 3, pos: 894
type: B, layer: 3, pos: 974
type: B, layer: 3, pos: 995
type: B, layer: 3, pos: 370
type: B, layer: 3, pos: 851
type: B, layer: 3, pos: 646
type: B, layer: 3, pos: 698
type: B, layer: 3, pos: 876
type: B, layer: 3, pos: 667
type: B, layer: 3, pos: 260
type: B, layer: 3, pos: 381
type: B, layer: 3, pos: 1019
type: B, layer: 3, pos: 673
type: B, layer: 3, pos: 235
type: B, layer: 3, pos: 316
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 419
type: B, layer: 3, pos: 867
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 980
type: B, layer: 3, pos: 967
type: B, layer: 3, pos: 319
type: B, layer: 3, pos: 315
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 258
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 376
type: B, layer: 3, pos: 883
type: B, layer: 3, pos: 259
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 700
type: B, layer: 3, pos: 836
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 1020
type: B, layer: 3, pos: 842
type: B, layer: 3, pos: 61
type: B, layer: 3, pos: 1018
type: B, layer: 3, pos: 1014
type: B, layer: 3, pos: 688
type: B, layer: 3, pos: 336
type: B, layer: 3, pos: 56
type: B, layer: 3, pos: 272
type: B, layer: 3, pos: 1010
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 51
type: B, layer: 3, pos: 345
type: B, layer: 3, pos: 656
type: B, layer: 3, pos: 340
type: B, layer: 3, pos: 975
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 657
type: B, layer: 3, pos: 201
type: B, layer: 3, pos: 360
type: B, layer: 3, pos: 645
type: B, layer: 3, pos: 690
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 649
type: B, layer: 3, pos: 683
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 220
type: B, layer: 3, pos: 335
type: B, layer: 3, pos: 703
type: B, layer: 3, pos: 62
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 102
type: B, layer: 3, pos: 870
type: B, layer: 3, pos: 344
type: B, layer: 3, pos: 68
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 337
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 111
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 404
type: B, layer: 3, pos: 651
type: B, layer: 3, pos: 1013
type: B, layer: 3, pos: 349
type: B, layer: 3, pos: 858
type: B, layer: 3, pos: 981
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 242
type: B, layer: 3, pos: 861
type: B, layer: 3, pos: 1004
type: B, layer: 3, pos: 279
type: B, layer: 3, pos: 987
type: B, layer: 3, pos: 325
type: B, layer: 3, pos: 281
type: B, layer: 3, pos: 113
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 78
type: B, layer: 3, pos: 658
type: B, layer: 3, pos: 57
type: B, layer: 3, pos: 54
type: B, layer: 3, pos: 297
type: B, layer: 3, pos: 203
type: B, layer: 3, pos: 835
type: B, layer: 3, pos: 971
type: B, layer: 3, pos: 420
type: B, layer: 3, pos: 63
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 55
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 879
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 595
type: B, layer: 3, pos: 263
type: B, layer: 3, pos: 969
type: B, layer: 3, pos: 642
type: B, layer: 3, pos: 702
type: B, layer: 3, pos: 318
type: B, layer: 3, pos: 863
type: B, layer: 3, pos: 983
type: B, layer: 3, pos: 328
type: B, layer: 3, pos: 257
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 50
type: B, layer: 3, pos: 343
type: B, layer: 3, pos: 965
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 365
type: B, layer: 3, pos: 855
type: B, layer: 3, pos: 428
type: B, layer: 3, pos: 664
type: B, layer: 3, pos: 86
type: B, layer: 3, pos: 246
type: B, layer: 3, pos: 354
type: B, layer: 3, pos: 598
type: B, layer: 3, pos: 252
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 643
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 238
type: B, layer: 3, pos: 333
type: B, layer: 3, pos: 264
type: B, layer: 3, pos: 1012
type: B, layer: 3, pos: 124
type: B, layer: 3, pos: 982
type: B, layer: 3, pos: 262
type: B, layer: 3, pos: 648
type: B, layer: 3, pos: 641
type: B, layer: 3, pos: 985
type: B, layer: 3, pos: 857
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 77
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 372
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 666
type: B, layer: 3, pos: 322
type: B, layer: 3, pos: 84
type: B, layer: 3, pos: 873
type: B, layer: 3, pos: 81
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 665
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 324
type: B, layer: 3, pos: 109
type: B, layer: 3, pos: 844
type: B, layer: 3, pos: 82
type: B, layer: 3, pos: 589
type: B, layer: 3, pos: 1007
type: B, layer: 3, pos: 663
type: B, layer: 3, pos: 696
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 209
type: B, layer: 3, pos: 94
type: B, layer: 3, pos: 296
type: B, layer: 3, pos: 989
type: B, layer: 3, pos: 251
type: B, layer: 3, pos: 885
type: B, layer: 3, pos: 990
type: B, layer: 3, pos: 104
type: B, layer: 3, pos: 681
type: B, layer: 3, pos: 261
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 358
type: B, layer: 3, pos: 304
type: B, layer: 3, pos: 127
type: B, layer: 3, pos: 280
type: B, layer: 3, pos: 628
type: B, layer: 3, pos: 249
type: B, layer: 3, pos: 853
type: B, layer: 3, pos: 421
type: B, layer: 3, pos: 833
type: B, layer: 3, pos: 674
type: B, layer: 3, pos: 207
type: B, layer: 3, pos: 986
type: B, layer: 3, pos: 610
type: B, layer: 3, pos: 123
type: B, layer: 3, pos: 998
type: B, layer: 3, pos: 847
type: B, layer: 3, pos: 53
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 597
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 672
type: B, layer: 3, pos: 270
type: B, layer: 3, pos: 590
type: B, layer: 3, pos: 202
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 282
type: B, layer: 3, pos: 368
type: B, layer: 3, pos: 596
type: B, layer: 3, pos: 1001
type: B, layer: 3, pos: 321
type: B, layer: 3, pos: 364
type: B, layer: 3, pos: 834
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 126
type: B, layer: 3, pos: 970
type: B, layer: 3, pos: 362
type: B, layer: 3, pos: 973
type: B, layer: 3, pos: 275
type: B, layer: 3, pos: 1002
type: B, layer: 3, pos: 617
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 630
type: B, layer: 3, pos: 352
type: B, layer: 3, pos: 205
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 97
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 1006
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 276
type: B, layer: 3, pos: 52
type: B, layer: 3, pos: 606
type: B, layer: 3, pos: 112
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 602
type: B, layer: 3, pos: 845
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 588
type: B, layer: 3, pos: 119
type: B, layer: 3, pos: 320
type: B, layer: 3, pos: 215
type: B, layer: 3, pos: 217
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 593
type: B, layer: 3, pos: 871
type: B, layer: 3, pos: 979
type: B, layer: 3, pos: 594
type: B, layer: 3, pos: 629
type: B, layer: 3, pos: 301
type: B, layer: 3, pos: 1022
type: B, layer: 3, pos: 866
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1008
type: B, layer: 3, pos: 89
type: B, layer: 3, pos: 114
type: B, layer: 3, pos: 87
type: B, layer: 3, pos: 862
type: B, layer: 3, pos: 587
type: B, layer: 3, pos: 580
type: B, layer: 3, pos: 88
type: B, layer: 3, pos: 966
type: B, layer: 3, pos: 210
type: B, layer: 3, pos: 852
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 960
type: B, layer: 3, pos: 694
type: B, layer: 3, pos: 692
type: B, layer: 3, pos: 371
type: B, layer: 3, pos: 1016
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 591
type: B, layer: 3, pos: 599
type: B, layer: 3, pos: 103
type: B, layer: 3, pos: 247
type: B, layer: 3, pos: 351
type: B, layer: 3, pos: 288
type: B, layer: 3, pos: 631
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 267
type: B, layer: 3, pos: 341
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 367
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 256
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 271
type: B, layer: 3, pos: 266
type: B, layer: 3, pos: 413
type: B, layer: 3, pos: 653
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 586
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 585
type: B, layer: 3, pos: 682
type: B, layer: 3, pos: 105
type: B, layer: 3, pos: 601
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 623
type: B, layer: 3, pos: 110
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 701
type: B, layer: 3, pos: 968
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 691
type: B, layer: 3, pos: 1011
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 685
type: B, layer: 3, pos: 125
type: B, layer: 3, pos: 639
type: B, layer: 3, pos: 650
type: B, layer: 3, pos: 74
type: B, layer: 3, pos: 687
type: B, layer: 3, pos: 680
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 269
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 303
type: B, layer: 3, pos: 618
type: B, layer: 3, pos: 285
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 622
type: B, layer: 3, pos: 609
type: B, layer: 3, pos: 636
type: B, layer: 3, pos: 405
type: B, layer: 3, pos: 607
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 243
type: B, layer: 3, pos: 626

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 292

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 228

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1051552, upper bound: 14.5023008
time: 58.63 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1082246, upper bound: 14.6102744
time: 67.73 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -37.5518837, -0.0870590, -37.5182228, -0.2904034, -37.2614822, 37.4311638
1: -17.5831947, 10.6131077, -17.5730782, 10.4741850, -28.0573807, 28.1861858
2: -14.3981857, 10.4919300, -14.3848591, 10.0816393, -24.4798241, 24.8767891
3: -14.8997536, 14.4316416, -14.8835783, 14.0667429, -28.9664955, 29.3152199
4: -15.1627789, 15.4909267, -15.0762167, 14.7478609, -29.9106407, 30.5671425
5: -14.2091503, 15.6173840, -14.1930933, 15.1864052, -29.3955555, 29.8104782
6: -21.2240448, 10.2550812, -20.8197994, 10.2650700, -31.4891148, 31.0748806
7: -17.4083595, 16.7097473, -17.3427048, 16.5195808, -33.5532227, 33.6830788
8: -16.3226433, 19.6721191, -16.2581730, 19.1571579, -35.4567413, 35.9101257
9: -15.2185097, 13.7180576, -15.1480083, 13.7103186, -28.6957397, 28.6785660
10: -24.0578880, 17.2562981, -23.5307426, 17.2027702, -41.2606583, 40.7870407
11: -27.0264072, 10.3127613, -26.2099285, 10.2798033, -37.3062096, 36.5226898
12: -25.1492558, 12.1401482, -24.2240219, 12.0582237, -37.2074814, 36.3641701
13: -22.2582016, 18.4617615, -22.1868439, 18.4073830, -40.6655846, 40.6486053
14: -48.2363586, -0.3918114, -47.8079453, -0.4343433, -47.6319275, 47.2243195
15: -19.5907822, 10.6944342, -19.5513973, 10.3160944, -29.9068756, 30.2458305
16: -25.2866497, 13.0665627, -24.9560528, 13.1333275, -37.9145470, 37.5707359
17: -44.6035500, 12.4331350, -43.9467850, 12.3822660, -55.6895905, 55.0844955
18: -20.4203739, 12.5232906, -20.3820877, 12.4727926, -32.8931656, 32.9053802
19: -18.2412853, 4.2624969, -17.9040718, 4.2634902, -22.5047760, 22.1665688
20: -15.3945608, 8.4762669, -15.2627439, 8.4657955, -23.8603554, 23.7390099
21: -26.3013077, 3.7739432, -25.8761902, 3.7624738, -30.0637817, 29.6501331
22: -32.9953117, -0.7890663, -32.8868294, -0.8882818, -30.8023911, 30.7028694
23: -18.2332458, 8.9168444, -17.9134159, 8.9072351, -27.1404800, 26.8302612
24: -25.2644424, 7.4187884, -25.2396126, 7.3370900, -31.1277580, 31.0747833
25: -18.3870049, 10.8842659, -18.2875938, 10.8162451, -29.2032509, 29.1718597
26: -23.9490891, 14.9270906, -23.6473274, 14.8856220, -38.8347092, 38.5744171
27: -26.2815361, 6.7291298, -26.2254658, 6.6817112, -32.0126343, 31.9128895
28: -17.5196896, 10.6170130, -17.3093109, 10.6172295, -27.9508667, 27.7336769
29: -40.4803619, -5.2730350, -40.1356964, -5.3011093, -34.2791672, 33.9424553
30: -21.0998535, 12.2974052, -20.8647118, 12.2963123, -33.3961639, 33.1621170
31: -23.9008484, 7.0047650, -23.6938133, 6.9871788, -30.8880272, 30.6985779
32: -27.9613533, 4.3550987, -27.6282883, 4.3463383, -31.4335976, 31.0900402
33: -30.5680389, 14.8998518, -30.4945793, 14.5873604, -44.2027435, 44.5468216
34: -26.0056095, 9.9879169, -25.9503384, 9.9032803, -35.9088898, 35.9382553
35: -27.7691822, 11.1247616, -27.7078743, 10.9579277, -38.2572403, 38.5054283
36: -27.2560749, 10.8664761, -27.1534576, 10.8793716, -37.7040787, 37.5795708
37: -37.3547058, 9.5503082, -37.1998787, 9.5885315, -45.6937866, 45.4736290
38: -29.7424488, 14.0583038, -29.6559105, 13.9854755, -43.7279243, 43.7142143
39: -38.5367813, 11.7669678, -38.4053650, 11.6145582, -49.5011444, 49.5012283
40: -30.4535370, 9.7118912, -30.3824692, 9.6906023, -38.4974327, 38.4417343
41: -22.6822453, 9.4380569, -22.3778858, 9.4721613, -32.1494293, 31.8159428
42: -17.0136051, 7.5940218, -16.3846455, 7.5394192, -24.1567554, 23.6400127

Time for backsubstitution: 0.94 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 292
type: B, layer: 3, pos: 228
type: B, layer: 3, pos: 356
type: B, layer: 3, pos: 229
type: B, layer: 3, pos: 868
type: B, layer: 3, pos: 355
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 357
type: B, layer: 3, pos: 363
type: B, layer: 3, pos: 348
type: B, layer: 3, pos: 284
type: B, layer: 3, pos: 869
type: B, layer: 3, pos: 997
type: B, layer: 3, pos: 887
type: B, layer: 3, pos: 377
type: B, layer: 3, pos: 353
type: B, layer: 3, pos: 369
type: B, layer: 3, pos: 375
type: B, layer: 3, pos: 875
type: B, layer: 3, pos: 892
type: B, layer: 3, pos: 293
type: B, layer: 3, pos: 988
type: B, layer: 3, pos: 999
type: B, layer: 3, pos: 881
type: B, layer: 3, pos: 291
type: B, layer: 3, pos: 378
type: B, layer: 3, pos: 991
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 305
type: B, layer: 3, pos: 283
type: B, layer: 3, pos: 996
type: B, layer: 3, pos: 289
type: B, layer: 3, pos: 383
type: B, layer: 3, pos: 993
type: B, layer: 3, pos: 380
type: B, layer: 3, pos: 1009
type: B, layer: 3, pos: 893
type: B, layer: 3, pos: 877
type: B, layer: 3, pos: 331
type: B, layer: 3, pos: 361
type: B, layer: 3, pos: 339
type: B, layer: 3, pos: 849
type: B, layer: 3, pos: 891
type: B, layer: 3, pos: 338
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 1015
type: B, layer: 3, pos: 972
type: B, layer: 3, pos: 850
type: B, layer: 3, pos: 676
type: B, layer: 3, pos: 231
type: B, layer: 3, pos: 889
type: B, layer: 3, pos: 865
type: B, layer: 3, pos: 843
type: B, layer: 3, pos: 684
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 300
type: B, layer: 3, pos: 859
type: B, layer: 3, pos: 895
type: B, layer: 3, pos: 882
type: B, layer: 3, pos: 347
type: B, layer: 3, pos: 689
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 1023
type: B, layer: 3, pos: 841
type: B, layer: 3, pos: 379
type: B, layer: 3, pos: 382
type: B, layer: 3, pos: 644
type: B, layer: 3, pos: 860
type: B, layer: 3, pos: 695
type: B, layer: 3, pos: 223
type: B, layer: 3, pos: 239
type: B, layer: 3, pos: 346
type: B, layer: 3, pos: 329
type: B, layer: 3, pos: 306
type: B, layer: 3, pos: 724
type: B, layer: 3, pos: 1003
type: B, layer: 3, pos: 265
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 313
type: B, layer: 3, pos: 334
type: B, layer: 3, pos: 273
type: B, layer: 3, pos: 314
type: B, layer: 3, pos: 85
type: B, layer: 3, pos: 978
type: B, layer: 3, pos: 874
type: B, layer: 3, pos: 1005
type: B, layer: 3, pos: 58
type: B, layer: 3, pos: 1021
type: B, layer: 3, pos: 846
type: B, layer: 3, pos: 69
type: B, layer: 3, pos: 884
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 647
type: B, layer: 3, pos: 1017
type: B, layer: 3, pos: 699
type: B, layer: 3, pos: 274
type: B, layer: 3, pos: 977
type: B, layer: 3, pos: 299
type: B, layer: 3, pos: 894
type: B, layer: 3, pos: 974
type: B, layer: 3, pos: 995
type: B, layer: 3, pos: 370
type: B, layer: 3, pos: 851
type: B, layer: 3, pos: 646
type: B, layer: 3, pos: 698
type: B, layer: 3, pos: 876
type: B, layer: 3, pos: 667
type: B, layer: 3, pos: 260
type: B, layer: 3, pos: 381
type: B, layer: 3, pos: 1019
type: B, layer: 3, pos: 673
type: B, layer: 3, pos: 235
type: B, layer: 3, pos: 316
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 419
type: B, layer: 3, pos: 867
type: B, layer: 3, pos: 204
type: B, layer: 3, pos: 980
type: B, layer: 3, pos: 967
type: B, layer: 3, pos: 319
type: B, layer: 3, pos: 315
type: B, layer: 3, pos: 697
type: B, layer: 3, pos: 258
type: B, layer: 3, pos: 214
type: B, layer: 3, pos: 376
type: B, layer: 3, pos: 883
type: B, layer: 3, pos: 259
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 700
type: B, layer: 3, pos: 836
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 1020
type: B, layer: 3, pos: 842
type: B, layer: 3, pos: 61
type: B, layer: 3, pos: 1018
type: B, layer: 3, pos: 1014
type: B, layer: 3, pos: 688
type: B, layer: 3, pos: 336
type: B, layer: 3, pos: 56
type: B, layer: 3, pos: 272
type: B, layer: 3, pos: 1010
type: B, layer: 3, pos: 255
type: B, layer: 3, pos: 51
type: B, layer: 3, pos: 345
type: B, layer: 3, pos: 656
type: B, layer: 3, pos: 340
type: B, layer: 3, pos: 975
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 657
type: B, layer: 3, pos: 201
type: B, layer: 3, pos: 360
type: B, layer: 3, pos: 645
type: B, layer: 3, pos: 690
type: B, layer: 3, pos: 604
type: B, layer: 3, pos: 649
type: B, layer: 3, pos: 683
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 220
type: B, layer: 3, pos: 335
type: B, layer: 3, pos: 703
type: B, layer: 3, pos: 62
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 102
type: B, layer: 3, pos: 870
type: B, layer: 3, pos: 344
type: B, layer: 3, pos: 68
type: B, layer: 3, pos: 71
type: B, layer: 3, pos: 337
type: B, layer: 3, pos: 326
type: B, layer: 3, pos: 655
type: B, layer: 3, pos: 111
type: B, layer: 3, pos: 59
type: B, layer: 3, pos: 404
type: B, layer: 3, pos: 651
type: B, layer: 3, pos: 1013
type: B, layer: 3, pos: 349
type: B, layer: 3, pos: 858
type: B, layer: 3, pos: 981
type: B, layer: 3, pos: 411
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 242
type: B, layer: 3, pos: 861
type: B, layer: 3, pos: 1004
type: B, layer: 3, pos: 279
type: B, layer: 3, pos: 987
type: B, layer: 3, pos: 325
type: B, layer: 3, pos: 281
type: B, layer: 3, pos: 113
type: B, layer: 3, pos: 317
type: B, layer: 3, pos: 95
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 233
type: B, layer: 3, pos: 225
type: B, layer: 3, pos: 78
type: B, layer: 3, pos: 658
type: B, layer: 3, pos: 57
type: B, layer: 3, pos: 54
type: B, layer: 3, pos: 297
type: B, layer: 3, pos: 203
type: B, layer: 3, pos: 835
type: B, layer: 3, pos: 971
type: B, layer: 3, pos: 420
type: B, layer: 3, pos: 63
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 55
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 879
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 595
type: B, layer: 3, pos: 263
type: B, layer: 3, pos: 969
type: B, layer: 3, pos: 642
type: B, layer: 3, pos: 702
type: B, layer: 3, pos: 318
type: B, layer: 3, pos: 863
type: B, layer: 3, pos: 983
type: B, layer: 3, pos: 328
type: B, layer: 3, pos: 257
type: B, layer: 3, pos: 675
type: B, layer: 3, pos: 50
type: B, layer: 3, pos: 343
type: B, layer: 3, pos: 965
type: B, layer: 3, pos: 206
type: B, layer: 3, pos: 365
type: B, layer: 3, pos: 855
type: B, layer: 3, pos: 428
type: B, layer: 3, pos: 664
type: B, layer: 3, pos: 86
type: B, layer: 3, pos: 246
type: B, layer: 3, pos: 354
type: B, layer: 3, pos: 598
type: B, layer: 3, pos: 252
type: B, layer: 3, pos: 677
type: B, layer: 3, pos: 643
type: B, layer: 3, pos: 227
type: B, layer: 3, pos: 238
type: B, layer: 3, pos: 333
type: B, layer: 3, pos: 264
type: B, layer: 3, pos: 1012
type: B, layer: 3, pos: 124
type: B, layer: 3, pos: 982
type: B, layer: 3, pos: 262
type: B, layer: 3, pos: 648
type: B, layer: 3, pos: 641
type: B, layer: 3, pos: 985
type: B, layer: 3, pos: 857
type: B, layer: 3, pos: 72
type: B, layer: 3, pos: 77
type: B, layer: 3, pos: 29
type: B, layer: 3, pos: 79
type: B, layer: 3, pos: 372
type: B, layer: 3, pos: 122
type: B, layer: 3, pos: 666
type: B, layer: 3, pos: 322
type: B, layer: 3, pos: 84
type: B, layer: 3, pos: 873
type: B, layer: 3, pos: 81
type: B, layer: 3, pos: 80
type: B, layer: 3, pos: 665
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 324
type: B, layer: 3, pos: 109
type: B, layer: 3, pos: 844
type: B, layer: 3, pos: 82
type: B, layer: 3, pos: 589
type: B, layer: 3, pos: 1007
type: B, layer: 3, pos: 663
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 696
type: B, layer: 3, pos: 209
type: B, layer: 3, pos: 94
type: B, layer: 3, pos: 296
type: B, layer: 3, pos: 989
type: B, layer: 3, pos: 251
type: B, layer: 3, pos: 885
type: B, layer: 3, pos: 990
type: B, layer: 3, pos: 104
type: B, layer: 3, pos: 681
type: B, layer: 3, pos: 261
type: B, layer: 3, pos: 34
type: B, layer: 3, pos: 358
type: B, layer: 3, pos: 304
type: B, layer: 3, pos: 127
type: B, layer: 3, pos: 280
type: B, layer: 3, pos: 628
type: B, layer: 3, pos: 249
type: B, layer: 3, pos: 853
type: B, layer: 3, pos: 421
type: B, layer: 3, pos: 833
type: B, layer: 3, pos: 674
type: B, layer: 3, pos: 207
type: B, layer: 3, pos: 986
type: B, layer: 3, pos: 610
type: B, layer: 3, pos: 123
type: B, layer: 3, pos: 998
type: B, layer: 3, pos: 847
type: B, layer: 3, pos: 53
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 597
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 672
type: B, layer: 3, pos: 270
type: B, layer: 3, pos: 590
type: B, layer: 3, pos: 202
type: B, layer: 3, pos: 250
type: B, layer: 3, pos: 282
type: B, layer: 3, pos: 368
type: B, layer: 3, pos: 596
type: B, layer: 3, pos: 1001
type: B, layer: 3, pos: 321
type: B, layer: 3, pos: 364
type: B, layer: 3, pos: 834
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 126
type: B, layer: 3, pos: 970
type: B, layer: 3, pos: 362
type: B, layer: 3, pos: 973
type: B, layer: 3, pos: 275
type: B, layer: 3, pos: 1002
type: B, layer: 3, pos: 617
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 630
type: B, layer: 3, pos: 352
type: B, layer: 3, pos: 205
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 97
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 1006
type: B, layer: 3, pos: 60
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 276
type: B, layer: 3, pos: 52
type: B, layer: 3, pos: 606
type: B, layer: 3, pos: 112
type: B, layer: 3, pos: 725
type: B, layer: 3, pos: 845
type: B, layer: 3, pos: 602
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 588
type: B, layer: 3, pos: 119
type: B, layer: 3, pos: 320
type: B, layer: 3, pos: 215
type: B, layer: 3, pos: 217
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 593
type: B, layer: 3, pos: 871
type: B, layer: 3, pos: 979
type: B, layer: 3, pos: 594
type: B, layer: 3, pos: 629
type: B, layer: 3, pos: 301
type: B, layer: 3, pos: 1022
type: B, layer: 3, pos: 866
type: B, layer: 3, pos: 654
type: B, layer: 3, pos: 1008
type: B, layer: 3, pos: 89
type: B, layer: 3, pos: 114
type: B, layer: 3, pos: 87
type: B, layer: 3, pos: 862
type: B, layer: 3, pos: 587
type: B, layer: 3, pos: 580
type: B, layer: 3, pos: 88
type: B, layer: 3, pos: 966
type: B, layer: 3, pos: 210
type: B, layer: 3, pos: 852
type: B, layer: 3, pos: 659
type: B, layer: 3, pos: 960
type: B, layer: 3, pos: 694
type: B, layer: 3, pos: 692
type: B, layer: 3, pos: 371
type: B, layer: 3, pos: 1016
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 591
type: B, layer: 3, pos: 599
type: B, layer: 3, pos: 103
type: B, layer: 3, pos: 247
type: B, layer: 3, pos: 351
type: B, layer: 3, pos: 288
type: B, layer: 3, pos: 631
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 267
type: B, layer: 3, pos: 341
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 367
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 256
type: B, layer: 3, pos: 627
type: B, layer: 3, pos: 271
type: B, layer: 3, pos: 266
type: B, layer: 3, pos: 413
type: B, layer: 3, pos: 653
type: B, layer: 3, pos: 234
type: B, layer: 3, pos: 652
type: B, layer: 3, pos: 586
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 585
type: B, layer: 3, pos: 682
type: B, layer: 3, pos: 105
type: B, layer: 3, pos: 601
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 623
type: B, layer: 3, pos: 110
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 701
type: B, layer: 3, pos: 968
type: B, layer: 3, pos: 23
type: B, layer: 3, pos: 691
type: B, layer: 3, pos: 9
type: B, layer: 3, pos: 1011
type: B, layer: 3, pos: 685
type: B, layer: 3, pos: 125
type: B, layer: 3, pos: 639
type: B, layer: 3, pos: 650
type: B, layer: 3, pos: 74
type: B, layer: 3, pos: 687
type: B, layer: 3, pos: 680
type: B, layer: 3, pos: 312
type: B, layer: 3, pos: 121
type: B, layer: 3, pos: 269
type: B, layer: 3, pos: 230
type: B, layer: 3, pos: 303
type: B, layer: 3, pos: 618
type: B, layer: 3, pos: 285
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 622
type: B, layer: 3, pos: 609
type: B, layer: 3, pos: 636
type: B, layer: 3, pos: 405
type: B, layer: 3, pos: 607
type: B, layer: 3, pos: 669
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 243
type: B, layer: 3, pos: 626

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 292

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 228

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1051552, upper bound: 14.5023008
time: 59.64 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1051552, upper bound: 14.6102744
time: 63.18 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 131.49 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 131.49
Output dim: 4, lower bound: -14.3605035, upper bound: 14.3620504
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 131.49
Output dim: 4, lower bound: -14.3605035, upper bound: 14.4673187
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 131.49
Output dim: 4, lower bound: -14.3605035, upper bound: 14.3620504
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 131.49
Output dim: 4, lower bound: -14.3605035, upper bound: 14.4673187
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 131.49
Output dim: 4, lower bound: -14.1596706, upper bound: 14.5212905
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 131.49
Output dim: 4, lower bound: -14.1596706, upper bound: 14.6366804
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 131.49
Output dim: 4, lower bound: -14.1596706, upper bound: 14.5212909
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 131.49
Output dim: 4, lower bound: -14.1596706, upper bound: 14.6366808
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 131.49
Output dim: 4, lower bound: -14.1051552, upper bound: 14.3698167
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 131.49
Output dim: 4, lower bound: -14.1082246, upper bound: 14.4897330
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 131.49
Output dim: 4, lower bound: -14.1051552, upper bound: 14.3698167
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 131.49
Output dim: 4, lower bound: -14.1082246, upper bound: 14.4897330
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 131.49
Output dim: 4, lower bound: -14.1051552, upper bound: 14.5023008
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 131.49
Output dim: 4, lower bound: -14.1082246, upper bound: 14.6102744
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 131.49
Output dim: 4, lower bound: -14.1051552, upper bound: 14.5023008
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 131.49
Output dim: 4, lower bound: -14.1051552, upper bound: 14.6102744

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 66.68 + 1852.60 = 1919.27 seconds
