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
execution time: IAR + RelationalAnalysis = 2.77 + 63.06 = 65.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -14.6721677, upper bound: 14.6721678

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 356
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 229
type: RSZ, layer: 3, pos: 868
type: RSZ, layer: 3, pos: 292
type: RSZ, layer: 3, pos: 236
type: RSZ, layer: 3, pos: 228
type: RSZ, layer: 3, pos: 355
type: RSZ, layer: 3, pos: 357
type: RSZ, layer: 3, pos: 284
type: RSZ, layer: 3, pos: 997
type: RSZ, layer: 3, pos: 724
type: RSZ, layer: 3, pos: 380
type: RSZ, layer: 3, pos: 363
type: RSZ, layer: 3, pos: 892
type: RSZ, layer: 3, pos: 375
type: RSZ, layer: 3, pos: 353
type: RSZ, layer: 3, pos: 377
type: RSZ, layer: 3, pos: 293
type: RSZ, layer: 3, pos: 887
type: RSZ, layer: 3, pos: 348
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 869
type: RSZ, layer: 3, pos: 316
type: RSZ, layer: 3, pos: 875
type: RSZ, layer: 3, pos: 378
type: RSZ, layer: 3, pos: 289
type: RSZ, layer: 3, pos: 369
type: RSZ, layer: 3, pos: 893
type: RSZ, layer: 3, pos: 351
type: RSZ, layer: 3, pos: 991
type: RSZ, layer: 3, pos: 999
type: RSZ, layer: 3, pos: 988
type: RSZ, layer: 3, pos: 305
type: RSZ, layer: 3, pos: 871
type: RSZ, layer: 3, pos: 881
type: RSZ, layer: 3, pos: 996
type: RSZ, layer: 3, pos: 876
type: RSZ, layer: 3, pos: 383
type: RSZ, layer: 3, pos: 299
type: RSZ, layer: 3, pos: 361
type: RSZ, layer: 3, pos: 877
type: RSZ, layer: 3, pos: 1015
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 381
type: RSZ, layer: 3, pos: 291
type: RSZ, layer: 3, pos: 306
type: RSZ, layer: 3, pos: 993
type: RSZ, layer: 3, pos: 382
type: RSZ, layer: 3, pos: 851
type: RSZ, layer: 3, pos: 334
type: RSZ, layer: 3, pos: 843
type: RSZ, layer: 3, pos: 865
type: RSZ, layer: 3, pos: 314
type: RSZ, layer: 3, pos: 282
type: RSZ, layer: 3, pos: 858
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 859
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 889
type: RSZ, layer: 3, pos: 863
type: RSZ, layer: 3, pos: 895
type: RSZ, layer: 3, pos: 1009
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 274
type: RSZ, layer: 3, pos: 346
type: RSZ, layer: 3, pos: 283
type: RSZ, layer: 3, pos: 849
type: RSZ, layer: 3, pos: 1004
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 271
type: RSZ, layer: 3, pos: 379
type: RSZ, layer: 3, pos: 1023
type: RSZ, layer: 3, pos: 972
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 335
type: RSZ, layer: 3, pos: 850
type: RSZ, layer: 3, pos: 986
type: RSZ, layer: 3, pos: 279
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 700
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 980
type: RSZ, layer: 3, pos: 1005
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 884
type: RSZ, layer: 3, pos: 340
type: RSZ, layer: 3, pos: 1003
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 343
type: RSZ, layer: 3, pos: 846
type: RSZ, layer: 3, pos: 684
type: RSZ, layer: 3, pos: 689
type: RSZ, layer: 3, pos: 885
type: RSZ, layer: 3, pos: 315
type: RSZ, layer: 3, pos: 1020
type: RSZ, layer: 3, pos: 978
type: RSZ, layer: 3, pos: 995
type: RSZ, layer: 3, pos: 300
type: RSZ, layer: 3, pos: 331
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 319
type: RSZ, layer: 3, pos: 882
type: RSZ, layer: 3, pos: 364
type: RSZ, layer: 3, pos: 370
type: RSZ, layer: 3, pos: 84
type: RSZ, layer: 3, pos: 1021
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 673
type: RSZ, layer: 3, pos: 235
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 338
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 63
type: RSZ, layer: 3, pos: 699
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 58
type: RSZ, layer: 3, pos: 85
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 297
type: RSZ, layer: 3, pos: 239
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 339
type: RSZ, layer: 3, pos: 281
type: RSZ, layer: 3, pos: 69
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1019
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 344
type: RSZ, layer: 3, pos: 68
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 270
type: RSZ, layer: 3, pos: 883
type: RSZ, layer: 3, pos: 358
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 372
type: RSZ, layer: 3, pos: 666
type: RSZ, layer: 3, pos: 974
type: RSZ, layer: 3, pos: 667
type: RSZ, layer: 3, pos: 644
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 656
type: RSZ, layer: 3, pos: 870
type: RSZ, layer: 3, pos: 690
type: RSZ, layer: 3, pos: 273
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 336
type: RSZ, layer: 3, pos: 647
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 860
type: RSZ, layer: 3, pos: 841
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 318
type: RSZ, layer: 3, pos: 977
type: RSZ, layer: 3, pos: 61
type: RSZ, layer: 3, pos: 879
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 345
type: RSZ, layer: 3, pos: 663
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 110
type: RSZ, layer: 3, pos: 1017
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 102
type: RSZ, layer: 3, pos: 967
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 688
type: RSZ, layer: 3, pos: 50
type: RSZ, layer: 3, pos: 214
type: RSZ, layer: 3, pos: 867
type: RSZ, layer: 3, pos: 1014
type: RSZ, layer: 3, pos: 652
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 354
type: RSZ, layer: 3, pos: 114
type: RSZ, layer: 3, pos: 1012
type: RSZ, layer: 3, pos: 1018
type: RSZ, layer: 3, pos: 329
type: RSZ, layer: 3, pos: 844
type: RSZ, layer: 3, pos: 703
type: RSZ, layer: 3, pos: 326
type: RSZ, layer: 3, pos: 201
type: RSZ, layer: 3, pos: 376
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 1010
type: RSZ, layer: 3, pos: 259
type: RSZ, layer: 3, pos: 371
type: RSZ, layer: 3, pos: 874
type: RSZ, layer: 3, pos: 57
type: RSZ, layer: 3, pos: 272
type: RSZ, layer: 3, pos: 347
type: RSZ, layer: 3, pos: 692
type: RSZ, layer: 3, pos: 646
type: RSZ, layer: 3, pos: 275
type: RSZ, layer: 3, pos: 56
type: RSZ, layer: 3, pos: 203
type: RSZ, layer: 3, pos: 657
type: RSZ, layer: 3, pos: 420
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 276
type: RSZ, layer: 3, pos: 113
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 265
type: RSZ, layer: 3, pos: 419
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 210
type: RSZ, layer: 3, pos: 324
type: RSZ, layer: 3, pos: 123
type: RSZ, layer: 3, pos: 285
type: RSZ, layer: 3, pos: 965
type: RSZ, layer: 3, pos: 303
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 643
type: RSZ, layer: 3, pos: 847
type: RSZ, layer: 3, pos: 94
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 985
type: RSZ, layer: 3, pos: 658
type: RSZ, layer: 3, pos: 362
type: RSZ, layer: 3, pos: 665
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 674
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 360
type: RSZ, layer: 3, pos: 702
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 54
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 982
type: RSZ, layer: 3, pos: 866
type: RSZ, layer: 3, pos: 852
type: RSZ, layer: 3, pos: 1013
type: RSZ, layer: 3, pos: 51
type: RSZ, layer: 3, pos: 989
type: RSZ, layer: 3, pos: 833
type: RSZ, layer: 3, pos: 664
type: RSZ, layer: 3, pos: 82
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 628
type: RSZ, layer: 3, pos: 694
type: RSZ, layer: 3, pos: 53
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 349
type: RSZ, layer: 3, pos: 258
type: RSZ, layer: 3, pos: 428
type: RSZ, layer: 3, pos: 333
type: RSZ, layer: 3, pos: 105
type: RSZ, layer: 3, pos: 971
type: RSZ, layer: 3, pos: 365
type: RSZ, layer: 3, pos: 202
type: RSZ, layer: 3, pos: 266
type: RSZ, layer: 3, pos: 651
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 595
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 681
type: RSZ, layer: 3, pos: 683
type: RSZ, layer: 3, pos: 645
type: RSZ, layer: 3, pos: 421
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 296
type: RSZ, layer: 3, pos: 845
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 280
type: RSZ, layer: 3, pos: 1006
type: RSZ, layer: 3, pos: 1007
type: RSZ, layer: 3, pos: 648
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 970
type: RSZ, layer: 3, pos: 260
type: RSZ, layer: 3, pos: 857
type: RSZ, layer: 3, pos: 341
type: RSZ, layer: 3, pos: 320
type: RSZ, layer: 3, pos: 55
type: RSZ, layer: 3, pos: 973
type: RSZ, layer: 3, pos: 301
type: RSZ, layer: 3, pos: 328
type: RSZ, layer: 3, pos: 623
type: RSZ, layer: 3, pos: 78
type: RSZ, layer: 3, pos: 650
type: RSZ, layer: 3, pos: 1011
type: RSZ, layer: 3, pos: 321
type: RSZ, layer: 3, pos: 598
type: RSZ, layer: 3, pos: 127
type: RSZ, layer: 3, pos: 207
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 597
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 649
type: RSZ, layer: 3, pos: 304
type: RSZ, layer: 3, pos: 596
type: RSZ, layer: 3, pos: 682
type: RSZ, layer: 3, pos: 701
type: RSZ, layer: 3, pos: 86
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 247
type: RSZ, layer: 3, pos: 109
type: RSZ, layer: 3, pos: 853
type: RSZ, layer: 3, pos: 1002
type: RSZ, layer: 3, pos: 261
type: RSZ, layer: 3, pos: 672
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 862
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 855
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 687
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 593
type: RSZ, layer: 3, pos: 243
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 205
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 1008
type: RSZ, layer: 3, pos: 630
type: RSZ, layer: 3, pos: 126
type: RSZ, layer: 3, pos: 602
type: RSZ, layer: 3, pos: 215
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 589
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 60
type: RSZ, layer: 3, pos: 367
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 835
type: RSZ, layer: 3, pos: 642
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 586
type: RSZ, layer: 3, pos: 609
type: RSZ, layer: 3, pos: 352
type: RSZ, layer: 3, pos: 263
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 696
type: RSZ, layer: 3, pos: 590
type: RSZ, layer: 3, pos: 1022
type: RSZ, layer: 3, pos: 587
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 288
type: RSZ, layer: 3, pos: 842
type: RSZ, layer: 3, pos: 327
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 836
type: RSZ, layer: 3, pos: 990
type: RSZ, layer: 3, pos: 269
type: RSZ, layer: 3, pos: 52
type: RSZ, layer: 3, pos: 981
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 264
type: RSZ, layer: 3, pos: 631
type: RSZ, layer: 3, pos: 636
type: RSZ, layer: 3, pos: 256
type: RSZ, layer: 3, pos: 966
type: RSZ, layer: 3, pos: 246
type: RSZ, layer: 3, pos: 873
type: RSZ, layer: 3, pos: 968
type: RSZ, layer: 3, pos: 594
type: RSZ, layer: 3, pos: 413
type: RSZ, layer: 3, pos: 622
type: RSZ, layer: 3, pos: 119
type: RSZ, layer: 3, pos: 1016
type: RSZ, layer: 3, pos: 653
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 639
type: RSZ, layer: 3, pos: 585
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 987
type: RSZ, layer: 3, pos: 606
type: RSZ, layer: 3, pos: 979
type: RSZ, layer: 3, pos: 405
type: RSZ, layer: 3, pos: 998
type: RSZ, layer: 3, pos: 641
type: RSZ, layer: 3, pos: 257
type: RSZ, layer: 3, pos: 267
type: RSZ, layer: 3, pos: 217
type: RSZ, layer: 3, pos: 680
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 618
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 617
type: RSZ, layer: 3, pos: 1001
type: RSZ, layer: 3, pos: 834
type: RSZ, layer: 3, pos: 97
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 685
type: RSZ, layer: 3, pos: 125
type: RSZ, layer: 3, pos: 983
type: RSZ, layer: 3, pos: 591
type: RSZ, layer: 3, pos: 368
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 87
type: RSZ, layer: 3, pos: 607
type: RSZ, layer: 3, pos: 88
type: RSZ, layer: 3, pos: 322
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 960
type: RSZ, layer: 3, pos: 691
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 861
type: RSZ, layer: 3, pos: 89
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 74
type: RSZ, layer: 3, pos: 629
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 262
type: RSZ, layer: 3, pos: 112
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 659
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 209
type: RSZ, layer: 3, pos: 601
type: RSZ, layer: 3, pos: 588
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 580
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 3, pos: 356

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.2515759, upper bound: 14.6653834
time: 68.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.6653834, upper bound: 14.2515759
time: 63.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 132.37 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 132.37
Output dim: 4, lower bound: -14.2515759, upper bound: 14.6653834
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 132.37
Output dim: 4, lower bound: -14.6653834, upper bound: 14.2515759

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -37.5578117, -0.2815094, -37.5578117, -0.2815094, -37.2763023, 37.2763023
1: -17.6220856, 10.4812832, -17.6220856, 10.4812832, -28.1033688, 28.1033688
2: -14.4312439, 10.0883579, -14.4312439, 10.0883579, -24.5196018, 24.5196018
3: -14.9140787, 14.0766926, -14.9140787, 14.0766926, -28.9907722, 28.9907722
4: -15.1334782, 14.7552061, -15.1334782, 14.7552061, -29.8886833, 29.8886833
5: -14.2266178, 15.1941137, -14.2266178, 15.1941137, -29.4207306, 29.4207306
6: -20.8351784, 10.3227959, -20.8351784, 10.3227959, -31.1579742, 31.1579742
7: -17.3737793, 16.5286179, -17.3737793, 16.5286179, -33.5201111, 33.5210266
8: -16.3228550, 19.1706161, -16.3228550, 19.1706161, -35.4593430, 35.4586639
9: -15.1635761, 13.7782021, -15.1635761, 13.7782021, -28.7458191, 28.7453041
10: -23.5488243, 17.2546597, -23.5488243, 17.2546597, -40.8034821, 40.8034821
11: -26.2301464, 10.3602743, -26.2301464, 10.3602743, -36.5904198, 36.5904198
12: -24.2345219, 12.1104794, -24.2345219, 12.1104794, -36.3450012, 36.3450012
13: -22.1990852, 18.4545364, -22.1990852, 18.4545364, -40.6536217, 40.6536217
14: -47.8707161, -0.4218502, -47.8707161, -0.4218502, -47.2839279, 47.2539215
15: -19.6529541, 10.3302298, -19.6529541, 10.3302298, -29.9831848, 29.9831848
16: -24.9793587, 13.2617989, -24.9793587, 13.2617989, -37.7791290, 37.7782440
17: -43.9613876, 12.4236822, -43.9613876, 12.4236822, -55.0834885, 55.0574875
18: -20.4577484, 12.4874125, -20.4577484, 12.4874125, -32.9451599, 32.9451599
19: -17.9183464, 4.2855158, -17.9183464, 4.2855158, -22.2038612, 22.2038612
20: -15.2772102, 8.4781389, -15.2772102, 8.4781389, -23.7553482, 23.7553482
21: -25.8919182, 3.7857480, -25.8919182, 3.7857480, -29.6776657, 29.6776657
22: -32.9530487, -0.8713036, -32.9530487, -0.8713036, -30.7347565, 30.7398605
23: -17.9241028, 8.9438848, -17.9241028, 8.9438848, -26.8679886, 26.8679886
24: -25.2896576, 7.3427725, -25.2896576, 7.3427725, -31.1361275, 31.1389618
25: -18.3243713, 10.8286724, -18.3243713, 10.8286724, -29.1530437, 29.1530437
26: -23.7112923, 14.9050426, -23.7112923, 14.9050426, -38.6163330, 38.6163330
27: -26.2949791, 6.6883535, -26.2949791, 6.6883535, -31.9727974, 31.9882698
28: -17.3191872, 10.6421642, -17.3191872, 10.6421642, -27.7748871, 27.7742958
29: -40.1622734, -5.2830868, -40.1622734, -5.2830868, -33.9363022, 33.9377289
30: -20.8790817, 12.3533125, -20.8790817, 12.3533125, -33.2323952, 33.2323952
31: -23.7122688, 7.0026283, -23.7122688, 7.0026283, -30.7148972, 30.7148972
32: -27.6387272, 4.3835154, -27.6387272, 4.3835154, -31.1176872, 31.1228600
33: -30.5414295, 14.6008110, -30.5414295, 14.6008110, -44.2399902, 44.2416420
34: -25.9748363, 9.9279261, -25.9748363, 9.9279261, -35.9027634, 35.9027634
35: -27.7439537, 10.9692993, -27.7439537, 10.9692993, -38.3036270, 38.3028526
36: -27.1807117, 10.9039993, -27.1807117, 10.9039993, -37.6403503, 37.6429291
37: -37.2227592, 9.6494160, -37.2227592, 9.6494160, -45.5747147, 45.5780640
38: -29.7080078, 13.9970264, -29.7080078, 13.9970264, -43.7050323, 43.7050323
39: -38.4577255, 11.6211214, -38.4577255, 11.6211214, -49.4293060, 49.4300308
40: -30.4051437, 9.7967424, -30.4051437, 9.7967424, -38.5557175, 38.5560226
41: -22.3953133, 9.5435400, -22.3953133, 9.5435400, -31.9388542, 31.9388542
42: -16.3960018, 7.5732532, -16.3960018, 7.5732532, -23.6386681, 23.6387882

Time for backsubstitution: 0.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 229
type: RSZ, layer: 3, pos: 868
type: RSZ, layer: 3, pos: 292
type: RSZ, layer: 3, pos: 236
type: RSZ, layer: 3, pos: 228
type: RSZ, layer: 3, pos: 355
type: RSZ, layer: 3, pos: 357
type: RSZ, layer: 3, pos: 284
type: RSZ, layer: 3, pos: 997
type: RSZ, layer: 3, pos: 724
type: RSZ, layer: 3, pos: 380
type: RSZ, layer: 3, pos: 363
type: RSZ, layer: 3, pos: 892
type: RSZ, layer: 3, pos: 375
type: RSZ, layer: 3, pos: 353
type: RSZ, layer: 3, pos: 377
type: RSZ, layer: 3, pos: 293
type: RSZ, layer: 3, pos: 887
type: RSZ, layer: 3, pos: 348
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 869
type: RSZ, layer: 3, pos: 316
type: RSZ, layer: 3, pos: 875
type: RSZ, layer: 3, pos: 378
type: RSZ, layer: 3, pos: 289
type: RSZ, layer: 3, pos: 369
type: RSZ, layer: 3, pos: 893
type: RSZ, layer: 3, pos: 351
type: RSZ, layer: 3, pos: 991
type: RSZ, layer: 3, pos: 999
type: RSZ, layer: 3, pos: 988
type: RSZ, layer: 3, pos: 305
type: RSZ, layer: 3, pos: 871
type: RSZ, layer: 3, pos: 881
type: RSZ, layer: 3, pos: 996
type: RSZ, layer: 3, pos: 876
type: RSZ, layer: 3, pos: 383
type: RSZ, layer: 3, pos: 299
type: RSZ, layer: 3, pos: 361
type: RSZ, layer: 3, pos: 877
type: RSZ, layer: 3, pos: 1015
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 381
type: RSZ, layer: 3, pos: 291
type: RSZ, layer: 3, pos: 306
type: RSZ, layer: 3, pos: 993
type: RSZ, layer: 3, pos: 382
type: RSZ, layer: 3, pos: 851
type: RSZ, layer: 3, pos: 334
type: RSZ, layer: 3, pos: 843
type: RSZ, layer: 3, pos: 865
type: RSZ, layer: 3, pos: 314
type: RSZ, layer: 3, pos: 282
type: RSZ, layer: 3, pos: 858
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 859
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 889
type: RSZ, layer: 3, pos: 863
type: RSZ, layer: 3, pos: 895
type: RSZ, layer: 3, pos: 1009
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 274
type: RSZ, layer: 3, pos: 346
type: RSZ, layer: 3, pos: 283
type: RSZ, layer: 3, pos: 849
type: RSZ, layer: 3, pos: 1004
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 271
type: RSZ, layer: 3, pos: 379
type: RSZ, layer: 3, pos: 1023
type: RSZ, layer: 3, pos: 972
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 335
type: RSZ, layer: 3, pos: 850
type: RSZ, layer: 3, pos: 986
type: RSZ, layer: 3, pos: 279
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 700
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 980
type: RSZ, layer: 3, pos: 1005
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 884
type: RSZ, layer: 3, pos: 340
type: RSZ, layer: 3, pos: 1003
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 343
type: RSZ, layer: 3, pos: 846
type: RSZ, layer: 3, pos: 684
type: RSZ, layer: 3, pos: 689
type: RSZ, layer: 3, pos: 885
type: RSZ, layer: 3, pos: 315
type: RSZ, layer: 3, pos: 1020
type: RSZ, layer: 3, pos: 978
type: RSZ, layer: 3, pos: 995
type: RSZ, layer: 3, pos: 300
type: RSZ, layer: 3, pos: 331
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 319
type: RSZ, layer: 3, pos: 882
type: RSZ, layer: 3, pos: 364
type: RSZ, layer: 3, pos: 370
type: RSZ, layer: 3, pos: 84
type: RSZ, layer: 3, pos: 1021
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 673
type: RSZ, layer: 3, pos: 235
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 338
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 63
type: RSZ, layer: 3, pos: 699
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 58
type: RSZ, layer: 3, pos: 85
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 297
type: RSZ, layer: 3, pos: 239
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 339
type: RSZ, layer: 3, pos: 281
type: RSZ, layer: 3, pos: 69
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1019
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 344
type: RSZ, layer: 3, pos: 68
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 270
type: RSZ, layer: 3, pos: 883
type: RSZ, layer: 3, pos: 358
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 372
type: RSZ, layer: 3, pos: 666
type: RSZ, layer: 3, pos: 974
type: RSZ, layer: 3, pos: 667
type: RSZ, layer: 3, pos: 644
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 656
type: RSZ, layer: 3, pos: 870
type: RSZ, layer: 3, pos: 690
type: RSZ, layer: 3, pos: 273
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 336
type: RSZ, layer: 3, pos: 647
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 860
type: RSZ, layer: 3, pos: 841
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 318
type: RSZ, layer: 3, pos: 977
type: RSZ, layer: 3, pos: 61
type: RSZ, layer: 3, pos: 879
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 345
type: RSZ, layer: 3, pos: 663
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 110
type: RSZ, layer: 3, pos: 1017
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 102
type: RSZ, layer: 3, pos: 967
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 688
type: RSZ, layer: 3, pos: 50
type: RSZ, layer: 3, pos: 214
type: RSZ, layer: 3, pos: 867
type: RSZ, layer: 3, pos: 1014
type: RSZ, layer: 3, pos: 652
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 354
type: RSZ, layer: 3, pos: 114
type: RSZ, layer: 3, pos: 1012
type: RSZ, layer: 3, pos: 1018
type: RSZ, layer: 3, pos: 329
type: RSZ, layer: 3, pos: 844
type: RSZ, layer: 3, pos: 703
type: RSZ, layer: 3, pos: 326
type: RSZ, layer: 3, pos: 201
type: RSZ, layer: 3, pos: 376
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 1010
type: RSZ, layer: 3, pos: 259
type: RSZ, layer: 3, pos: 371
type: RSZ, layer: 3, pos: 874
type: RSZ, layer: 3, pos: 57
type: RSZ, layer: 3, pos: 272
type: RSZ, layer: 3, pos: 347
type: RSZ, layer: 3, pos: 692
type: RSZ, layer: 3, pos: 646
type: RSZ, layer: 3, pos: 275
type: RSZ, layer: 3, pos: 56
type: RSZ, layer: 3, pos: 203
type: RSZ, layer: 3, pos: 657
type: RSZ, layer: 3, pos: 420
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 276
type: RSZ, layer: 3, pos: 113
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 265
type: RSZ, layer: 3, pos: 419
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 210
type: RSZ, layer: 3, pos: 324
type: RSZ, layer: 3, pos: 123
type: RSZ, layer: 3, pos: 285
type: RSZ, layer: 3, pos: 965
type: RSZ, layer: 3, pos: 303
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 643
type: RSZ, layer: 3, pos: 847
type: RSZ, layer: 3, pos: 94
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 985
type: RSZ, layer: 3, pos: 658
type: RSZ, layer: 3, pos: 362
type: RSZ, layer: 3, pos: 665
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 674
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 360
type: RSZ, layer: 3, pos: 702
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 54
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 982
type: RSZ, layer: 3, pos: 866
type: RSZ, layer: 3, pos: 852
type: RSZ, layer: 3, pos: 1013
type: RSZ, layer: 3, pos: 51
type: RSZ, layer: 3, pos: 989
type: RSZ, layer: 3, pos: 833
type: RSZ, layer: 3, pos: 664
type: RSZ, layer: 3, pos: 82
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 628
type: RSZ, layer: 3, pos: 694
type: RSZ, layer: 3, pos: 53
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 349
type: RSZ, layer: 3, pos: 258
type: RSZ, layer: 3, pos: 428
type: RSZ, layer: 3, pos: 333
type: RSZ, layer: 3, pos: 105
type: RSZ, layer: 3, pos: 971
type: RSZ, layer: 3, pos: 365
type: RSZ, layer: 3, pos: 202
type: RSZ, layer: 3, pos: 266
type: RSZ, layer: 3, pos: 651
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 595
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 681
type: RSZ, layer: 3, pos: 683
type: RSZ, layer: 3, pos: 645
type: RSZ, layer: 3, pos: 421
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 296
type: RSZ, layer: 3, pos: 845
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 280
type: RSZ, layer: 3, pos: 1006
type: RSZ, layer: 3, pos: 1007
type: RSZ, layer: 3, pos: 648
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 970
type: RSZ, layer: 3, pos: 260
type: RSZ, layer: 3, pos: 857
type: RSZ, layer: 3, pos: 341
type: RSZ, layer: 3, pos: 320
type: RSZ, layer: 3, pos: 55
type: RSZ, layer: 3, pos: 973
type: RSZ, layer: 3, pos: 301
type: RSZ, layer: 3, pos: 328
type: RSZ, layer: 3, pos: 623
type: RSZ, layer: 3, pos: 78
type: RSZ, layer: 3, pos: 650
type: RSZ, layer: 3, pos: 1011
type: RSZ, layer: 3, pos: 321
type: RSZ, layer: 3, pos: 598
type: RSZ, layer: 3, pos: 127
type: RSZ, layer: 3, pos: 207
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 597
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 649
type: RSZ, layer: 3, pos: 304
type: RSZ, layer: 3, pos: 596
type: RSZ, layer: 3, pos: 682
type: RSZ, layer: 3, pos: 701
type: RSZ, layer: 3, pos: 86
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 247
type: RSZ, layer: 3, pos: 109
type: RSZ, layer: 3, pos: 853
type: RSZ, layer: 3, pos: 1002
type: RSZ, layer: 3, pos: 261
type: RSZ, layer: 3, pos: 672
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 862
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 855
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 687
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 593
type: RSZ, layer: 3, pos: 243
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 205
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 1008
type: RSZ, layer: 3, pos: 630
type: RSZ, layer: 3, pos: 126
type: RSZ, layer: 3, pos: 602
type: RSZ, layer: 3, pos: 215
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 589
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 60
type: RSZ, layer: 3, pos: 367
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 835
type: RSZ, layer: 3, pos: 642
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 586
type: RSZ, layer: 3, pos: 609
type: RSZ, layer: 3, pos: 352
type: RSZ, layer: 3, pos: 263
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 696
type: RSZ, layer: 3, pos: 590
type: RSZ, layer: 3, pos: 1022
type: RSZ, layer: 3, pos: 587
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 288
type: RSZ, layer: 3, pos: 842
type: RSZ, layer: 3, pos: 327
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 836
type: RSZ, layer: 3, pos: 990
type: RSZ, layer: 3, pos: 269
type: RSZ, layer: 3, pos: 52
type: RSZ, layer: 3, pos: 981
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 264
type: RSZ, layer: 3, pos: 631
type: RSZ, layer: 3, pos: 636
type: RSZ, layer: 3, pos: 256
type: RSZ, layer: 3, pos: 966
type: RSZ, layer: 3, pos: 246
type: RSZ, layer: 3, pos: 873
type: RSZ, layer: 3, pos: 968
type: RSZ, layer: 3, pos: 594
type: RSZ, layer: 3, pos: 413
type: RSZ, layer: 3, pos: 622
type: RSZ, layer: 3, pos: 119
type: RSZ, layer: 3, pos: 1016
type: RSZ, layer: 3, pos: 653
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 639
type: RSZ, layer: 3, pos: 585
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 987
type: RSZ, layer: 3, pos: 606
type: RSZ, layer: 3, pos: 979
type: RSZ, layer: 3, pos: 405
type: RSZ, layer: 3, pos: 998
type: RSZ, layer: 3, pos: 641
type: RSZ, layer: 3, pos: 257
type: RSZ, layer: 3, pos: 267
type: RSZ, layer: 3, pos: 217
type: RSZ, layer: 3, pos: 680
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 618
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 617
type: RSZ, layer: 3, pos: 1001
type: RSZ, layer: 3, pos: 834
type: RSZ, layer: 3, pos: 97
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 685
type: RSZ, layer: 3, pos: 125
type: RSZ, layer: 3, pos: 983
type: RSZ, layer: 3, pos: 591
type: RSZ, layer: 3, pos: 368
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 87
type: RSZ, layer: 3, pos: 607
type: RSZ, layer: 3, pos: 88
type: RSZ, layer: 3, pos: 322
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 960
type: RSZ, layer: 3, pos: 691
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 861
type: RSZ, layer: 3, pos: 89
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 74
type: RSZ, layer: 3, pos: 629
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 262
type: RSZ, layer: 3, pos: 112
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 659
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 209
type: RSZ, layer: 3, pos: 601
type: RSZ, layer: 3, pos: 588
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 580
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 237

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1651007, upper bound: 14.6518672
time: 62.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.2316472, upper bound: 14.5367414
time: 55.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -37.5578117, -0.2815094, -37.5578117, -0.2815094, -37.2763023, 37.2763023
1: -17.6220856, 10.4812832, -17.6220856, 10.4812832, -28.1033688, 28.1033688
2: -14.4312439, 10.0883579, -14.4312439, 10.0883579, -24.5196018, 24.5196018
3: -14.9140787, 14.0766926, -14.9140787, 14.0766926, -28.9907722, 28.9907722
4: -15.1334782, 14.7552061, -15.1334782, 14.7552061, -29.8886833, 29.8886833
5: -14.2266178, 15.1941137, -14.2266178, 15.1941137, -29.4207306, 29.4207306
6: -20.8351784, 10.3227959, -20.8351784, 10.3227959, -31.1579742, 31.1579742
7: -17.3737793, 16.5286179, -17.3737793, 16.5286179, -33.5210266, 33.5201073
8: -16.3228550, 19.1706161, -16.3228550, 19.1706161, -35.4586639, 35.4593468
9: -15.1635761, 13.7782021, -15.1635761, 13.7782021, -28.7453003, 28.7458191
10: -23.5488243, 17.2546597, -23.5488243, 17.2546597, -40.8034821, 40.8034821
11: -26.2301464, 10.3602743, -26.2301464, 10.3602743, -36.5904198, 36.5904198
12: -24.2345219, 12.1104794, -24.2345219, 12.1104794, -36.3450012, 36.3450012
13: -22.1990852, 18.4545364, -22.1990852, 18.4545364, -40.6536217, 40.6536217
14: -47.8707161, -0.4218502, -47.8707161, -0.4218502, -47.2539215, 47.2839279
15: -19.6529541, 10.3302298, -19.6529541, 10.3302298, -29.9831848, 29.9831848
16: -24.9793587, 13.2617989, -24.9793587, 13.2617989, -37.7782440, 37.7791290
17: -43.9613876, 12.4236822, -43.9613876, 12.4236822, -55.0574875, 55.0834885
18: -20.4577484, 12.4874125, -20.4577484, 12.4874125, -32.9451599, 32.9451599
19: -17.9183464, 4.2855158, -17.9183464, 4.2855158, -22.2038612, 22.2038612
20: -15.2772102, 8.4781389, -15.2772102, 8.4781389, -23.7553482, 23.7553482
21: -25.8919182, 3.7857480, -25.8919182, 3.7857480, -29.6776657, 29.6776657
22: -32.9530487, -0.8713036, -32.9530487, -0.8713036, -30.7398682, 30.7347603
23: -17.9241028, 8.9438848, -17.9241028, 8.9438848, -26.8679886, 26.8679886
24: -25.2896576, 7.3427725, -25.2896576, 7.3427725, -31.1389656, 31.1361294
25: -18.3243713, 10.8286724, -18.3243713, 10.8286724, -29.1530437, 29.1530437
26: -23.7112923, 14.9050426, -23.7112923, 14.9050426, -38.6163330, 38.6163330
27: -26.2949791, 6.6883535, -26.2949791, 6.6883535, -31.9882698, 31.9728012
28: -17.3191872, 10.6421642, -17.3191872, 10.6421642, -27.7742920, 27.7748871
29: -40.1622734, -5.2830868, -40.1622734, -5.2830868, -33.9377289, 33.9363060
30: -20.8790817, 12.3533125, -20.8790817, 12.3533125, -33.2323952, 33.2323952
31: -23.7122688, 7.0026283, -23.7122688, 7.0026283, -30.7148972, 30.7148972
32: -27.6387272, 4.3835154, -27.6387272, 4.3835154, -31.1228600, 31.1176853
33: -30.5414295, 14.6008110, -30.5414295, 14.6008110, -44.2416382, 44.2399979
34: -25.9748363, 9.9279261, -25.9748363, 9.9279261, -35.9027634, 35.9027634
35: -27.7439537, 10.9692993, -27.7439537, 10.9692993, -38.3028488, 38.3036270
36: -27.1807117, 10.9039993, -27.1807117, 10.9039993, -37.6429291, 37.6403503
37: -37.2227592, 9.6494160, -37.2227592, 9.6494160, -45.5780640, 45.5747147
38: -29.7080078, 13.9970264, -29.7080078, 13.9970264, -43.7050323, 43.7050323
39: -38.4577255, 11.6211214, -38.4577255, 11.6211214, -49.4300308, 49.4292984
40: -30.4051437, 9.7967424, -30.4051437, 9.7967424, -38.5560226, 38.5557213
41: -22.3953133, 9.5435400, -22.3953133, 9.5435400, -31.9388542, 31.9388542
42: -16.3960018, 7.5732532, -16.3960018, 7.5732532, -23.6387863, 23.6386700

Time for backsubstitution: 0.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 237
type: RSZ, layer: 3, pos: 229
type: RSZ, layer: 3, pos: 868
type: RSZ, layer: 3, pos: 292
type: RSZ, layer: 3, pos: 236
type: RSZ, layer: 3, pos: 228
type: RSZ, layer: 3, pos: 355
type: RSZ, layer: 3, pos: 357
type: RSZ, layer: 3, pos: 284
type: RSZ, layer: 3, pos: 997
type: RSZ, layer: 3, pos: 724
type: RSZ, layer: 3, pos: 380
type: RSZ, layer: 3, pos: 363
type: RSZ, layer: 3, pos: 892
type: RSZ, layer: 3, pos: 375
type: RSZ, layer: 3, pos: 353
type: RSZ, layer: 3, pos: 377
type: RSZ, layer: 3, pos: 293
type: RSZ, layer: 3, pos: 887
type: RSZ, layer: 3, pos: 348
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 869
type: RSZ, layer: 3, pos: 316
type: RSZ, layer: 3, pos: 875
type: RSZ, layer: 3, pos: 378
type: RSZ, layer: 3, pos: 289
type: RSZ, layer: 3, pos: 369
type: RSZ, layer: 3, pos: 893
type: RSZ, layer: 3, pos: 351
type: RSZ, layer: 3, pos: 991
type: RSZ, layer: 3, pos: 999
type: RSZ, layer: 3, pos: 988
type: RSZ, layer: 3, pos: 305
type: RSZ, layer: 3, pos: 871
type: RSZ, layer: 3, pos: 881
type: RSZ, layer: 3, pos: 996
type: RSZ, layer: 3, pos: 876
type: RSZ, layer: 3, pos: 383
type: RSZ, layer: 3, pos: 299
type: RSZ, layer: 3, pos: 361
type: RSZ, layer: 3, pos: 877
type: RSZ, layer: 3, pos: 1015
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 381
type: RSZ, layer: 3, pos: 291
type: RSZ, layer: 3, pos: 306
type: RSZ, layer: 3, pos: 993
type: RSZ, layer: 3, pos: 382
type: RSZ, layer: 3, pos: 851
type: RSZ, layer: 3, pos: 334
type: RSZ, layer: 3, pos: 843
type: RSZ, layer: 3, pos: 865
type: RSZ, layer: 3, pos: 314
type: RSZ, layer: 3, pos: 282
type: RSZ, layer: 3, pos: 858
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 859
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 889
type: RSZ, layer: 3, pos: 863
type: RSZ, layer: 3, pos: 895
type: RSZ, layer: 3, pos: 1009
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 274
type: RSZ, layer: 3, pos: 346
type: RSZ, layer: 3, pos: 283
type: RSZ, layer: 3, pos: 849
type: RSZ, layer: 3, pos: 1004
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 271
type: RSZ, layer: 3, pos: 379
type: RSZ, layer: 3, pos: 1023
type: RSZ, layer: 3, pos: 972
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 335
type: RSZ, layer: 3, pos: 850
type: RSZ, layer: 3, pos: 986
type: RSZ, layer: 3, pos: 279
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 700
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 980
type: RSZ, layer: 3, pos: 1005
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 884
type: RSZ, layer: 3, pos: 340
type: RSZ, layer: 3, pos: 1003
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 343
type: RSZ, layer: 3, pos: 846
type: RSZ, layer: 3, pos: 684
type: RSZ, layer: 3, pos: 689
type: RSZ, layer: 3, pos: 885
type: RSZ, layer: 3, pos: 315
type: RSZ, layer: 3, pos: 1020
type: RSZ, layer: 3, pos: 978
type: RSZ, layer: 3, pos: 995
type: RSZ, layer: 3, pos: 300
type: RSZ, layer: 3, pos: 331
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 319
type: RSZ, layer: 3, pos: 882
type: RSZ, layer: 3, pos: 364
type: RSZ, layer: 3, pos: 370
type: RSZ, layer: 3, pos: 84
type: RSZ, layer: 3, pos: 1021
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 673
type: RSZ, layer: 3, pos: 235
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 338
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 63
type: RSZ, layer: 3, pos: 699
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 58
type: RSZ, layer: 3, pos: 85
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 297
type: RSZ, layer: 3, pos: 239
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 339
type: RSZ, layer: 3, pos: 281
type: RSZ, layer: 3, pos: 69
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1019
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 344
type: RSZ, layer: 3, pos: 68
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 270
type: RSZ, layer: 3, pos: 883
type: RSZ, layer: 3, pos: 358
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 372
type: RSZ, layer: 3, pos: 666
type: RSZ, layer: 3, pos: 974
type: RSZ, layer: 3, pos: 667
type: RSZ, layer: 3, pos: 644
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 656
type: RSZ, layer: 3, pos: 870
type: RSZ, layer: 3, pos: 690
type: RSZ, layer: 3, pos: 273
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 336
type: RSZ, layer: 3, pos: 647
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 860
type: RSZ, layer: 3, pos: 841
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 318
type: RSZ, layer: 3, pos: 977
type: RSZ, layer: 3, pos: 61
type: RSZ, layer: 3, pos: 879
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 345
type: RSZ, layer: 3, pos: 663
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 110
type: RSZ, layer: 3, pos: 1017
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 102
type: RSZ, layer: 3, pos: 967
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 688
type: RSZ, layer: 3, pos: 50
type: RSZ, layer: 3, pos: 214
type: RSZ, layer: 3, pos: 867
type: RSZ, layer: 3, pos: 1014
type: RSZ, layer: 3, pos: 652
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 354
type: RSZ, layer: 3, pos: 114
type: RSZ, layer: 3, pos: 1012
type: RSZ, layer: 3, pos: 1018
type: RSZ, layer: 3, pos: 329
type: RSZ, layer: 3, pos: 844
type: RSZ, layer: 3, pos: 703
type: RSZ, layer: 3, pos: 326
type: RSZ, layer: 3, pos: 201
type: RSZ, layer: 3, pos: 376
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 1010
type: RSZ, layer: 3, pos: 259
type: RSZ, layer: 3, pos: 371
type: RSZ, layer: 3, pos: 874
type: RSZ, layer: 3, pos: 57
type: RSZ, layer: 3, pos: 272
type: RSZ, layer: 3, pos: 347
type: RSZ, layer: 3, pos: 692
type: RSZ, layer: 3, pos: 646
type: RSZ, layer: 3, pos: 275
type: RSZ, layer: 3, pos: 56
type: RSZ, layer: 3, pos: 203
type: RSZ, layer: 3, pos: 657
type: RSZ, layer: 3, pos: 420
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 276
type: RSZ, layer: 3, pos: 113
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 265
type: RSZ, layer: 3, pos: 419
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 210
type: RSZ, layer: 3, pos: 324
type: RSZ, layer: 3, pos: 123
type: RSZ, layer: 3, pos: 285
type: RSZ, layer: 3, pos: 965
type: RSZ, layer: 3, pos: 303
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 643
type: RSZ, layer: 3, pos: 847
type: RSZ, layer: 3, pos: 94
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 985
type: RSZ, layer: 3, pos: 658
type: RSZ, layer: 3, pos: 362
type: RSZ, layer: 3, pos: 665
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 674
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 360
type: RSZ, layer: 3, pos: 702
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 54
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 982
type: RSZ, layer: 3, pos: 866
type: RSZ, layer: 3, pos: 852
type: RSZ, layer: 3, pos: 1013
type: RSZ, layer: 3, pos: 51
type: RSZ, layer: 3, pos: 989
type: RSZ, layer: 3, pos: 833
type: RSZ, layer: 3, pos: 664
type: RSZ, layer: 3, pos: 82
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 628
type: RSZ, layer: 3, pos: 694
type: RSZ, layer: 3, pos: 53
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 349
type: RSZ, layer: 3, pos: 258
type: RSZ, layer: 3, pos: 428
type: RSZ, layer: 3, pos: 333
type: RSZ, layer: 3, pos: 105
type: RSZ, layer: 3, pos: 971
type: RSZ, layer: 3, pos: 365
type: RSZ, layer: 3, pos: 202
type: RSZ, layer: 3, pos: 266
type: RSZ, layer: 3, pos: 651
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 595
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 681
type: RSZ, layer: 3, pos: 683
type: RSZ, layer: 3, pos: 645
type: RSZ, layer: 3, pos: 421
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 296
type: RSZ, layer: 3, pos: 845
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 280
type: RSZ, layer: 3, pos: 1006
type: RSZ, layer: 3, pos: 1007
type: RSZ, layer: 3, pos: 648
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 970
type: RSZ, layer: 3, pos: 260
type: RSZ, layer: 3, pos: 857
type: RSZ, layer: 3, pos: 341
type: RSZ, layer: 3, pos: 320
type: RSZ, layer: 3, pos: 55
type: RSZ, layer: 3, pos: 973
type: RSZ, layer: 3, pos: 301
type: RSZ, layer: 3, pos: 328
type: RSZ, layer: 3, pos: 623
type: RSZ, layer: 3, pos: 78
type: RSZ, layer: 3, pos: 650
type: RSZ, layer: 3, pos: 1011
type: RSZ, layer: 3, pos: 321
type: RSZ, layer: 3, pos: 598
type: RSZ, layer: 3, pos: 127
type: RSZ, layer: 3, pos: 207
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 597
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 649
type: RSZ, layer: 3, pos: 304
type: RSZ, layer: 3, pos: 596
type: RSZ, layer: 3, pos: 682
type: RSZ, layer: 3, pos: 701
type: RSZ, layer: 3, pos: 86
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 247
type: RSZ, layer: 3, pos: 109
type: RSZ, layer: 3, pos: 853
type: RSZ, layer: 3, pos: 1002
type: RSZ, layer: 3, pos: 261
type: RSZ, layer: 3, pos: 672
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 862
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 855
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 687
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 593
type: RSZ, layer: 3, pos: 243
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 205
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 1008
type: RSZ, layer: 3, pos: 630
type: RSZ, layer: 3, pos: 126
type: RSZ, layer: 3, pos: 602
type: RSZ, layer: 3, pos: 215
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 589
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 60
type: RSZ, layer: 3, pos: 367
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 835
type: RSZ, layer: 3, pos: 642
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 586
type: RSZ, layer: 3, pos: 609
type: RSZ, layer: 3, pos: 352
type: RSZ, layer: 3, pos: 263
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 696
type: RSZ, layer: 3, pos: 590
type: RSZ, layer: 3, pos: 1022
type: RSZ, layer: 3, pos: 587
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 288
type: RSZ, layer: 3, pos: 842
type: RSZ, layer: 3, pos: 327
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 836
type: RSZ, layer: 3, pos: 990
type: RSZ, layer: 3, pos: 269
type: RSZ, layer: 3, pos: 52
type: RSZ, layer: 3, pos: 981
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 264
type: RSZ, layer: 3, pos: 631
type: RSZ, layer: 3, pos: 636
type: RSZ, layer: 3, pos: 256
type: RSZ, layer: 3, pos: 966
type: RSZ, layer: 3, pos: 246
type: RSZ, layer: 3, pos: 873
type: RSZ, layer: 3, pos: 968
type: RSZ, layer: 3, pos: 594
type: RSZ, layer: 3, pos: 413
type: RSZ, layer: 3, pos: 622
type: RSZ, layer: 3, pos: 119
type: RSZ, layer: 3, pos: 1016
type: RSZ, layer: 3, pos: 653
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 639
type: RSZ, layer: 3, pos: 585
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 987
type: RSZ, layer: 3, pos: 606
type: RSZ, layer: 3, pos: 979
type: RSZ, layer: 3, pos: 405
type: RSZ, layer: 3, pos: 998
type: RSZ, layer: 3, pos: 641
type: RSZ, layer: 3, pos: 257
type: RSZ, layer: 3, pos: 267
type: RSZ, layer: 3, pos: 217
type: RSZ, layer: 3, pos: 680
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 618
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 617
type: RSZ, layer: 3, pos: 1001
type: RSZ, layer: 3, pos: 834
type: RSZ, layer: 3, pos: 97
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 685
type: RSZ, layer: 3, pos: 125
type: RSZ, layer: 3, pos: 983
type: RSZ, layer: 3, pos: 591
type: RSZ, layer: 3, pos: 368
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 87
type: RSZ, layer: 3, pos: 607
type: RSZ, layer: 3, pos: 88
type: RSZ, layer: 3, pos: 322
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 960
type: RSZ, layer: 3, pos: 691
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 861
type: RSZ, layer: 3, pos: 89
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 74
type: RSZ, layer: 3, pos: 629
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 262
type: RSZ, layer: 3, pos: 112
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 659
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 209
type: RSZ, layer: 3, pos: 601
type: RSZ, layer: 3, pos: 588
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 580
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 3, pos: 237

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.5367414, upper bound: 14.2316472
time: 61.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.6518672, upper bound: 14.1651007
time: 53.93 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 116.34 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 116.34
Output dim: 4, lower bound: -14.1651007, upper bound: 14.6518672
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 116.34
Output dim: 4, lower bound: -14.2316472, upper bound: 14.5367414
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 116.34
Output dim: 4, lower bound: -14.5367414, upper bound: 14.2316472
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 116.34
Output dim: 4, lower bound: -14.6518672, upper bound: 14.1651007

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.5578117, -0.2815094, -37.5578117, -0.2815094, -37.2763023, 37.2763023
1: -17.6220856, 10.4812832, -17.6220856, 10.4812832, -28.1033688, 28.1033688
2: -14.4312439, 10.0883579, -14.4312439, 10.0883579, -24.5196018, 24.5196018
3: -14.9140787, 14.0766926, -14.9140787, 14.0766926, -28.9907722, 28.9907722
4: -15.1334782, 14.7552061, -15.1334782, 14.7552061, -29.8886833, 29.8886833
5: -14.2266178, 15.1941137, -14.2266178, 15.1941137, -29.4207306, 29.4207306
6: -20.8351784, 10.3227959, -20.8351784, 10.3227959, -31.1579742, 31.1579742
7: -17.3737793, 16.5286179, -17.3737793, 16.5286179, -33.5176926, 33.5181084
8: -16.3228550, 19.1706161, -16.3228550, 19.1706161, -35.4586754, 35.4570961
9: -15.1635761, 13.7782021, -15.1635761, 13.7782021, -28.7455864, 28.7422504
10: -23.5488243, 17.2546597, -23.5488243, 17.2546597, -40.8034821, 40.8034821
11: -26.2301464, 10.3602743, -26.2301464, 10.3602743, -36.5904198, 36.5904198
12: -24.2345219, 12.1104794, -24.2345219, 12.1104794, -36.3450012, 36.3450012
13: -22.1990852, 18.4545364, -22.1990852, 18.4545364, -40.6536217, 40.6536217
14: -47.8707161, -0.4218502, -47.8707161, -0.4218502, -47.2761993, 47.2399139
15: -19.6529541, 10.3302298, -19.6529541, 10.3302298, -29.9831848, 29.9831848
16: -24.9793587, 13.2617989, -24.9793587, 13.2617989, -37.7778893, 37.7722397
17: -43.9613876, 12.4236822, -43.9613876, 12.4236822, -55.0764313, 55.0456085
18: -20.4577484, 12.4874125, -20.4577484, 12.4874125, -32.9451599, 32.9451599
19: -17.9183464, 4.2855158, -17.9183464, 4.2855158, -22.2038612, 22.2038612
20: -15.2772102, 8.4781389, -15.2772102, 8.4781389, -23.7553482, 23.7553482
21: -25.8919182, 3.7857480, -25.8919182, 3.7857480, -29.6776657, 29.6776657
22: -32.9530487, -0.8713036, -32.9530487, -0.8713036, -30.7225647, 30.7358856
23: -17.9241028, 8.9438848, -17.9241028, 8.9438848, -26.8679886, 26.8679886
24: -25.2896576, 7.3427725, -25.2896576, 7.3427725, -31.1214180, 31.1336708
25: -18.3243713, 10.8286724, -18.3243713, 10.8286724, -29.1530437, 29.1530437
26: -23.7112923, 14.9050426, -23.7112923, 14.9050426, -38.6163330, 38.6163330
27: -26.2949791, 6.6883535, -26.2949791, 6.6883535, -31.9595604, 31.9810066
28: -17.3191872, 10.6421642, -17.3191872, 10.6421642, -27.7739029, 27.7728348
29: -40.1622734, -5.2830868, -40.1622734, -5.2830868, -33.9326401, 33.9400597
30: -20.8790817, 12.3533125, -20.8790817, 12.3533125, -33.2323952, 33.2323952
31: -23.7122688, 7.0026283, -23.7122688, 7.0026283, -30.7148972, 30.7148972
32: -27.6387272, 4.3835154, -27.6387272, 4.3835154, -31.1131897, 31.1194172
33: -30.5414295, 14.6008110, -30.5414295, 14.6008110, -44.2355423, 44.2378845
34: -25.9748363, 9.9279261, -25.9748363, 9.9279261, -35.9027634, 35.9027634
35: -27.7439537, 10.9692993, -27.7439537, 10.9692993, -38.2997742, 38.2993240
36: -27.1807117, 10.9039993, -27.1807117, 10.9039993, -37.6389618, 37.6421852
37: -37.2227592, 9.6494160, -37.2227592, 9.6494160, -45.5719757, 45.5767517
38: -29.7080078, 13.9970264, -29.7080078, 13.9970264, -43.7050323, 43.7050323
39: -38.4577255, 11.6211214, -38.4577255, 11.6211214, -49.4263077, 49.4283905
40: -30.4051437, 9.7967424, -30.4051437, 9.7967424, -38.5542679, 38.5552216
41: -22.3953133, 9.5435400, -22.3953133, 9.5435400, -31.9388542, 31.9361458
42: -16.3960018, 7.5732532, -16.3960018, 7.5732532, -23.6216011, 23.6145000

Time for backsubstitution: 0.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 229
type: RSZ, layer: 3, pos: 868
type: RSZ, layer: 3, pos: 292
type: RSZ, layer: 3, pos: 236
type: RSZ, layer: 3, pos: 228
type: RSZ, layer: 3, pos: 355
type: RSZ, layer: 3, pos: 357
type: RSZ, layer: 3, pos: 284
type: RSZ, layer: 3, pos: 997
type: RSZ, layer: 3, pos: 724
type: RSZ, layer: 3, pos: 380
type: RSZ, layer: 3, pos: 363
type: RSZ, layer: 3, pos: 892
type: RSZ, layer: 3, pos: 375
type: RSZ, layer: 3, pos: 353
type: RSZ, layer: 3, pos: 377
type: RSZ, layer: 3, pos: 293
type: RSZ, layer: 3, pos: 887
type: RSZ, layer: 3, pos: 348
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 869
type: RSZ, layer: 3, pos: 316
type: RSZ, layer: 3, pos: 875
type: RSZ, layer: 3, pos: 378
type: RSZ, layer: 3, pos: 289
type: RSZ, layer: 3, pos: 369
type: RSZ, layer: 3, pos: 893
type: RSZ, layer: 3, pos: 351
type: RSZ, layer: 3, pos: 991
type: RSZ, layer: 3, pos: 999
type: RSZ, layer: 3, pos: 988
type: RSZ, layer: 3, pos: 305
type: RSZ, layer: 3, pos: 871
type: RSZ, layer: 3, pos: 881
type: RSZ, layer: 3, pos: 996
type: RSZ, layer: 3, pos: 876
type: RSZ, layer: 3, pos: 383
type: RSZ, layer: 3, pos: 299
type: RSZ, layer: 3, pos: 361
type: RSZ, layer: 3, pos: 877
type: RSZ, layer: 3, pos: 1015
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 381
type: RSZ, layer: 3, pos: 291
type: RSZ, layer: 3, pos: 306
type: RSZ, layer: 3, pos: 993
type: RSZ, layer: 3, pos: 382
type: RSZ, layer: 3, pos: 851
type: RSZ, layer: 3, pos: 334
type: RSZ, layer: 3, pos: 843
type: RSZ, layer: 3, pos: 865
type: RSZ, layer: 3, pos: 314
type: RSZ, layer: 3, pos: 282
type: RSZ, layer: 3, pos: 858
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 859
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 889
type: RSZ, layer: 3, pos: 863
type: RSZ, layer: 3, pos: 895
type: RSZ, layer: 3, pos: 1009
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 274
type: RSZ, layer: 3, pos: 346
type: RSZ, layer: 3, pos: 283
type: RSZ, layer: 3, pos: 849
type: RSZ, layer: 3, pos: 1004
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 271
type: RSZ, layer: 3, pos: 379
type: RSZ, layer: 3, pos: 1023
type: RSZ, layer: 3, pos: 972
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 335
type: RSZ, layer: 3, pos: 850
type: RSZ, layer: 3, pos: 986
type: RSZ, layer: 3, pos: 279
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 700
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 980
type: RSZ, layer: 3, pos: 1005
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 884
type: RSZ, layer: 3, pos: 340
type: RSZ, layer: 3, pos: 1003
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 343
type: RSZ, layer: 3, pos: 846
type: RSZ, layer: 3, pos: 684
type: RSZ, layer: 3, pos: 689
type: RSZ, layer: 3, pos: 885
type: RSZ, layer: 3, pos: 315
type: RSZ, layer: 3, pos: 1020
type: RSZ, layer: 3, pos: 978
type: RSZ, layer: 3, pos: 995
type: RSZ, layer: 3, pos: 300
type: RSZ, layer: 3, pos: 331
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 319
type: RSZ, layer: 3, pos: 882
type: RSZ, layer: 3, pos: 364
type: RSZ, layer: 3, pos: 370
type: RSZ, layer: 3, pos: 84
type: RSZ, layer: 3, pos: 1021
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 673
type: RSZ, layer: 3, pos: 235
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 338
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 63
type: RSZ, layer: 3, pos: 699
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 58
type: RSZ, layer: 3, pos: 85
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 297
type: RSZ, layer: 3, pos: 239
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 339
type: RSZ, layer: 3, pos: 281
type: RSZ, layer: 3, pos: 69
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1019
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 344
type: RSZ, layer: 3, pos: 68
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 270
type: RSZ, layer: 3, pos: 883
type: RSZ, layer: 3, pos: 358
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 372
type: RSZ, layer: 3, pos: 666
type: RSZ, layer: 3, pos: 974
type: RSZ, layer: 3, pos: 667
type: RSZ, layer: 3, pos: 644
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 656
type: RSZ, layer: 3, pos: 870
type: RSZ, layer: 3, pos: 690
type: RSZ, layer: 3, pos: 273
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 336
type: RSZ, layer: 3, pos: 647
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 860
type: RSZ, layer: 3, pos: 841
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 318
type: RSZ, layer: 3, pos: 977
type: RSZ, layer: 3, pos: 61
type: RSZ, layer: 3, pos: 879
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 345
type: RSZ, layer: 3, pos: 663
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 110
type: RSZ, layer: 3, pos: 1017
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 102
type: RSZ, layer: 3, pos: 967
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 688
type: RSZ, layer: 3, pos: 50
type: RSZ, layer: 3, pos: 214
type: RSZ, layer: 3, pos: 867
type: RSZ, layer: 3, pos: 1014
type: RSZ, layer: 3, pos: 652
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 354
type: RSZ, layer: 3, pos: 114
type: RSZ, layer: 3, pos: 1012
type: RSZ, layer: 3, pos: 1018
type: RSZ, layer: 3, pos: 329
type: RSZ, layer: 3, pos: 844
type: RSZ, layer: 3, pos: 703
type: RSZ, layer: 3, pos: 326
type: RSZ, layer: 3, pos: 201
type: RSZ, layer: 3, pos: 376
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 1010
type: RSZ, layer: 3, pos: 259
type: RSZ, layer: 3, pos: 371
type: RSZ, layer: 3, pos: 874
type: RSZ, layer: 3, pos: 57
type: RSZ, layer: 3, pos: 272
type: RSZ, layer: 3, pos: 347
type: RSZ, layer: 3, pos: 692
type: RSZ, layer: 3, pos: 646
type: RSZ, layer: 3, pos: 275
type: RSZ, layer: 3, pos: 56
type: RSZ, layer: 3, pos: 203
type: RSZ, layer: 3, pos: 657
type: RSZ, layer: 3, pos: 420
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 276
type: RSZ, layer: 3, pos: 113
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 265
type: RSZ, layer: 3, pos: 419
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 210
type: RSZ, layer: 3, pos: 324
type: RSZ, layer: 3, pos: 123
type: RSZ, layer: 3, pos: 285
type: RSZ, layer: 3, pos: 965
type: RSZ, layer: 3, pos: 303
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 643
type: RSZ, layer: 3, pos: 847
type: RSZ, layer: 3, pos: 94
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 985
type: RSZ, layer: 3, pos: 658
type: RSZ, layer: 3, pos: 362
type: RSZ, layer: 3, pos: 665
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 674
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 360
type: RSZ, layer: 3, pos: 702
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 54
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 982
type: RSZ, layer: 3, pos: 866
type: RSZ, layer: 3, pos: 852
type: RSZ, layer: 3, pos: 1013
type: RSZ, layer: 3, pos: 51
type: RSZ, layer: 3, pos: 989
type: RSZ, layer: 3, pos: 833
type: RSZ, layer: 3, pos: 664
type: RSZ, layer: 3, pos: 82
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 628
type: RSZ, layer: 3, pos: 694
type: RSZ, layer: 3, pos: 53
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 349
type: RSZ, layer: 3, pos: 258
type: RSZ, layer: 3, pos: 428
type: RSZ, layer: 3, pos: 333
type: RSZ, layer: 3, pos: 105
type: RSZ, layer: 3, pos: 971
type: RSZ, layer: 3, pos: 365
type: RSZ, layer: 3, pos: 202
type: RSZ, layer: 3, pos: 266
type: RSZ, layer: 3, pos: 651
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 595
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 681
type: RSZ, layer: 3, pos: 683
type: RSZ, layer: 3, pos: 645
type: RSZ, layer: 3, pos: 421
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 296
type: RSZ, layer: 3, pos: 845
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 280
type: RSZ, layer: 3, pos: 1006
type: RSZ, layer: 3, pos: 1007
type: RSZ, layer: 3, pos: 648
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 970
type: RSZ, layer: 3, pos: 260
type: RSZ, layer: 3, pos: 857
type: RSZ, layer: 3, pos: 341
type: RSZ, layer: 3, pos: 320
type: RSZ, layer: 3, pos: 55
type: RSZ, layer: 3, pos: 973
type: RSZ, layer: 3, pos: 301
type: RSZ, layer: 3, pos: 328
type: RSZ, layer: 3, pos: 623
type: RSZ, layer: 3, pos: 78
type: RSZ, layer: 3, pos: 650
type: RSZ, layer: 3, pos: 1011
type: RSZ, layer: 3, pos: 321
type: RSZ, layer: 3, pos: 598
type: RSZ, layer: 3, pos: 127
type: RSZ, layer: 3, pos: 207
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 597
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 649
type: RSZ, layer: 3, pos: 304
type: RSZ, layer: 3, pos: 596
type: RSZ, layer: 3, pos: 682
type: RSZ, layer: 3, pos: 701
type: RSZ, layer: 3, pos: 86
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 247
type: RSZ, layer: 3, pos: 109
type: RSZ, layer: 3, pos: 853
type: RSZ, layer: 3, pos: 1002
type: RSZ, layer: 3, pos: 261
type: RSZ, layer: 3, pos: 672
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 862
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 855
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 687
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 593
type: RSZ, layer: 3, pos: 243
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 205
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 1008
type: RSZ, layer: 3, pos: 630
type: RSZ, layer: 3, pos: 126
type: RSZ, layer: 3, pos: 602
type: RSZ, layer: 3, pos: 215
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 589
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 60
type: RSZ, layer: 3, pos: 367
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 835
type: RSZ, layer: 3, pos: 642
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 586
type: RSZ, layer: 3, pos: 609
type: RSZ, layer: 3, pos: 352
type: RSZ, layer: 3, pos: 263
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 696
type: RSZ, layer: 3, pos: 590
type: RSZ, layer: 3, pos: 1022
type: RSZ, layer: 3, pos: 587
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 288
type: RSZ, layer: 3, pos: 842
type: RSZ, layer: 3, pos: 327
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 836
type: RSZ, layer: 3, pos: 990
type: RSZ, layer: 3, pos: 269
type: RSZ, layer: 3, pos: 52
type: RSZ, layer: 3, pos: 981
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 264
type: RSZ, layer: 3, pos: 631
type: RSZ, layer: 3, pos: 636
type: RSZ, layer: 3, pos: 256
type: RSZ, layer: 3, pos: 966
type: RSZ, layer: 3, pos: 246
type: RSZ, layer: 3, pos: 873
type: RSZ, layer: 3, pos: 968
type: RSZ, layer: 3, pos: 594
type: RSZ, layer: 3, pos: 413
type: RSZ, layer: 3, pos: 622
type: RSZ, layer: 3, pos: 119
type: RSZ, layer: 3, pos: 1016
type: RSZ, layer: 3, pos: 653
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 639
type: RSZ, layer: 3, pos: 585
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 987
type: RSZ, layer: 3, pos: 606
type: RSZ, layer: 3, pos: 979
type: RSZ, layer: 3, pos: 405
type: RSZ, layer: 3, pos: 998
type: RSZ, layer: 3, pos: 641
type: RSZ, layer: 3, pos: 257
type: RSZ, layer: 3, pos: 267
type: RSZ, layer: 3, pos: 217
type: RSZ, layer: 3, pos: 680
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 618
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 617
type: RSZ, layer: 3, pos: 1001
type: RSZ, layer: 3, pos: 834
type: RSZ, layer: 3, pos: 97
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 685
type: RSZ, layer: 3, pos: 125
type: RSZ, layer: 3, pos: 983
type: RSZ, layer: 3, pos: 591
type: RSZ, layer: 3, pos: 368
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 87
type: RSZ, layer: 3, pos: 607
type: RSZ, layer: 3, pos: 88
type: RSZ, layer: 3, pos: 322
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 960
type: RSZ, layer: 3, pos: 691
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 861
type: RSZ, layer: 3, pos: 89
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 74
type: RSZ, layer: 3, pos: 629
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 262
type: RSZ, layer: 3, pos: 112
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 659
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 209
type: RSZ, layer: 3, pos: 601
type: RSZ, layer: 3, pos: 588
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 580
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.0397480, upper bound: 14.6363273
time: 66.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1553625, upper bound: 14.5159429
time: 57.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.5578117, -0.2815094, -37.5578117, -0.2815094, -37.2763023, 37.2763023
1: -17.6220856, 10.4812832, -17.6220856, 10.4812832, -28.1033688, 28.1033688
2: -14.4312439, 10.0883579, -14.4312439, 10.0883579, -24.5196018, 24.5196018
3: -14.9140787, 14.0766926, -14.9140787, 14.0766926, -28.9907722, 28.9907722
4: -15.1334782, 14.7552061, -15.1334782, 14.7552061, -29.8886833, 29.8886833
5: -14.2266178, 15.1941137, -14.2266178, 15.1941137, -29.4207306, 29.4207306
6: -20.8351784, 10.3227959, -20.8351784, 10.3227959, -31.1579742, 31.1579742
7: -17.3737793, 16.5286179, -17.3737793, 16.5286179, -33.5169296, 33.5186157
8: -16.3228550, 19.1706161, -16.3228550, 19.1706161, -35.4577751, 35.4579468
9: -15.1635761, 13.7782021, -15.1635761, 13.7782021, -28.7427635, 28.7448750
10: -23.5488243, 17.2546597, -23.5488243, 17.2546597, -40.8034821, 40.8034821
11: -26.2301464, 10.3602743, -26.2301464, 10.3602743, -36.5904198, 36.5904198
12: -24.2345219, 12.1104794, -24.2345219, 12.1104794, -36.3450012, 36.3450012
13: -22.1990852, 18.4545364, -22.1990852, 18.4545364, -40.6536217, 40.6536217
14: -47.8707161, -0.4218502, -47.8707161, -0.4218502, -47.2699203, 47.2460022
15: -19.6529541, 10.3302298, -19.6529541, 10.3302298, -29.9831848, 29.9831848
16: -24.9793587, 13.2617989, -24.9793587, 13.2617989, -37.7731285, 37.7769547
17: -43.9613876, 12.4236822, -43.9613876, 12.4236822, -55.0716248, 55.0502167
18: -20.4577484, 12.4874125, -20.4577484, 12.4874125, -32.9451599, 32.9451599
19: -17.9183464, 4.2855158, -17.9183464, 4.2855158, -22.2038612, 22.2038612
20: -15.2772102, 8.4781389, -15.2772102, 8.4781389, -23.7553482, 23.7553482
21: -25.8919182, 3.7857480, -25.8919182, 3.7857480, -29.6776657, 29.6776657
22: -32.9530487, -0.8713036, -32.9530487, -0.8713036, -30.7307281, 30.7276688
23: -17.9241028, 8.9438848, -17.9241028, 8.9438848, -26.8679886, 26.8679886
24: -25.2896576, 7.3427725, -25.2896576, 7.3427725, -31.1306190, 31.1242466
25: -18.3243713, 10.8286724, -18.3243713, 10.8286724, -29.1530437, 29.1530437
26: -23.7112923, 14.9050426, -23.7112923, 14.9050426, -38.6163330, 38.6163330
27: -26.2949791, 6.6883535, -26.2949791, 6.6883535, -31.9650154, 31.9750328
28: -17.3191872, 10.6421642, -17.3191872, 10.6421642, -27.7734222, 27.7733135
29: -40.1622734, -5.2830868, -40.1622734, -5.2830868, -33.9386368, 33.9340591
30: -20.8790817, 12.3533125, -20.8790817, 12.3533125, -33.2323952, 33.2323952
31: -23.7122688, 7.0026283, -23.7122688, 7.0026283, -30.7148972, 30.7148972
32: -27.6387272, 4.3835154, -27.6387272, 4.3835154, -31.1142426, 31.1183624
33: -30.5414295, 14.6008110, -30.5414295, 14.6008110, -44.2362442, 44.2369652
34: -25.9748363, 9.9279261, -25.9748363, 9.9279261, -35.9027634, 35.9027634
35: -27.7439537, 10.9692993, -27.7439537, 10.9692993, -38.3000946, 38.2985420
36: -27.1807117, 10.9039993, -27.1807117, 10.9039993, -37.6395950, 37.6415329
37: -37.2227592, 9.6494160, -37.2227592, 9.6494160, -45.5734100, 45.5753174
38: -29.7080078, 13.9970264, -29.7080078, 13.9970264, -43.7050323, 43.7050323
39: -38.4577255, 11.6211214, -38.4577255, 11.6211214, -49.4276581, 49.4270401
40: -30.4051437, 9.7967424, -30.4051437, 9.7967424, -38.5549240, 38.5545769
41: -22.3953133, 9.5435400, -22.3953133, 9.5435400, -31.9361801, 31.9388542
42: -16.3960018, 7.5732532, -16.3960018, 7.5732532, -23.6143799, 23.6198330

Time for backsubstitution: 0.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 229
type: RSZ, layer: 3, pos: 868
type: RSZ, layer: 3, pos: 292
type: RSZ, layer: 3, pos: 236
type: RSZ, layer: 3, pos: 228
type: RSZ, layer: 3, pos: 355
type: RSZ, layer: 3, pos: 357
type: RSZ, layer: 3, pos: 284
type: RSZ, layer: 3, pos: 997
type: RSZ, layer: 3, pos: 724
type: RSZ, layer: 3, pos: 380
type: RSZ, layer: 3, pos: 363
type: RSZ, layer: 3, pos: 892
type: RSZ, layer: 3, pos: 375
type: RSZ, layer: 3, pos: 353
type: RSZ, layer: 3, pos: 377
type: RSZ, layer: 3, pos: 293
type: RSZ, layer: 3, pos: 887
type: RSZ, layer: 3, pos: 348
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 869
type: RSZ, layer: 3, pos: 316
type: RSZ, layer: 3, pos: 875
type: RSZ, layer: 3, pos: 378
type: RSZ, layer: 3, pos: 289
type: RSZ, layer: 3, pos: 369
type: RSZ, layer: 3, pos: 893
type: RSZ, layer: 3, pos: 351
type: RSZ, layer: 3, pos: 991
type: RSZ, layer: 3, pos: 999
type: RSZ, layer: 3, pos: 988
type: RSZ, layer: 3, pos: 305
type: RSZ, layer: 3, pos: 871
type: RSZ, layer: 3, pos: 881
type: RSZ, layer: 3, pos: 996
type: RSZ, layer: 3, pos: 876
type: RSZ, layer: 3, pos: 383
type: RSZ, layer: 3, pos: 299
type: RSZ, layer: 3, pos: 361
type: RSZ, layer: 3, pos: 877
type: RSZ, layer: 3, pos: 1015
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 381
type: RSZ, layer: 3, pos: 291
type: RSZ, layer: 3, pos: 306
type: RSZ, layer: 3, pos: 993
type: RSZ, layer: 3, pos: 382
type: RSZ, layer: 3, pos: 851
type: RSZ, layer: 3, pos: 334
type: RSZ, layer: 3, pos: 843
type: RSZ, layer: 3, pos: 865
type: RSZ, layer: 3, pos: 314
type: RSZ, layer: 3, pos: 282
type: RSZ, layer: 3, pos: 858
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 859
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 889
type: RSZ, layer: 3, pos: 863
type: RSZ, layer: 3, pos: 895
type: RSZ, layer: 3, pos: 1009
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 274
type: RSZ, layer: 3, pos: 346
type: RSZ, layer: 3, pos: 283
type: RSZ, layer: 3, pos: 849
type: RSZ, layer: 3, pos: 1004
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 271
type: RSZ, layer: 3, pos: 379
type: RSZ, layer: 3, pos: 1023
type: RSZ, layer: 3, pos: 972
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 335
type: RSZ, layer: 3, pos: 850
type: RSZ, layer: 3, pos: 986
type: RSZ, layer: 3, pos: 279
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 700
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 980
type: RSZ, layer: 3, pos: 1005
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 884
type: RSZ, layer: 3, pos: 340
type: RSZ, layer: 3, pos: 1003
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 343
type: RSZ, layer: 3, pos: 846
type: RSZ, layer: 3, pos: 684
type: RSZ, layer: 3, pos: 689
type: RSZ, layer: 3, pos: 885
type: RSZ, layer: 3, pos: 315
type: RSZ, layer: 3, pos: 1020
type: RSZ, layer: 3, pos: 978
type: RSZ, layer: 3, pos: 995
type: RSZ, layer: 3, pos: 300
type: RSZ, layer: 3, pos: 331
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 319
type: RSZ, layer: 3, pos: 882
type: RSZ, layer: 3, pos: 364
type: RSZ, layer: 3, pos: 370
type: RSZ, layer: 3, pos: 84
type: RSZ, layer: 3, pos: 1021
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 673
type: RSZ, layer: 3, pos: 235
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 338
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 63
type: RSZ, layer: 3, pos: 699
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 58
type: RSZ, layer: 3, pos: 85
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 297
type: RSZ, layer: 3, pos: 239
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 339
type: RSZ, layer: 3, pos: 281
type: RSZ, layer: 3, pos: 69
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1019
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 344
type: RSZ, layer: 3, pos: 68
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 270
type: RSZ, layer: 3, pos: 883
type: RSZ, layer: 3, pos: 358
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 372
type: RSZ, layer: 3, pos: 666
type: RSZ, layer: 3, pos: 974
type: RSZ, layer: 3, pos: 667
type: RSZ, layer: 3, pos: 644
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 656
type: RSZ, layer: 3, pos: 870
type: RSZ, layer: 3, pos: 690
type: RSZ, layer: 3, pos: 273
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 336
type: RSZ, layer: 3, pos: 647
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 860
type: RSZ, layer: 3, pos: 841
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 318
type: RSZ, layer: 3, pos: 977
type: RSZ, layer: 3, pos: 61
type: RSZ, layer: 3, pos: 879
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 345
type: RSZ, layer: 3, pos: 663
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 110
type: RSZ, layer: 3, pos: 1017
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 102
type: RSZ, layer: 3, pos: 967
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 688
type: RSZ, layer: 3, pos: 50
type: RSZ, layer: 3, pos: 214
type: RSZ, layer: 3, pos: 867
type: RSZ, layer: 3, pos: 1014
type: RSZ, layer: 3, pos: 652
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 354
type: RSZ, layer: 3, pos: 114
type: RSZ, layer: 3, pos: 1012
type: RSZ, layer: 3, pos: 1018
type: RSZ, layer: 3, pos: 329
type: RSZ, layer: 3, pos: 844
type: RSZ, layer: 3, pos: 703
type: RSZ, layer: 3, pos: 326
type: RSZ, layer: 3, pos: 201
type: RSZ, layer: 3, pos: 376
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 1010
type: RSZ, layer: 3, pos: 259
type: RSZ, layer: 3, pos: 371
type: RSZ, layer: 3, pos: 874
type: RSZ, layer: 3, pos: 57
type: RSZ, layer: 3, pos: 272
type: RSZ, layer: 3, pos: 347
type: RSZ, layer: 3, pos: 692
type: RSZ, layer: 3, pos: 646
type: RSZ, layer: 3, pos: 275
type: RSZ, layer: 3, pos: 56
type: RSZ, layer: 3, pos: 203
type: RSZ, layer: 3, pos: 657
type: RSZ, layer: 3, pos: 420
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 276
type: RSZ, layer: 3, pos: 113
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 265
type: RSZ, layer: 3, pos: 419
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 210
type: RSZ, layer: 3, pos: 324
type: RSZ, layer: 3, pos: 123
type: RSZ, layer: 3, pos: 285
type: RSZ, layer: 3, pos: 965
type: RSZ, layer: 3, pos: 303
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 643
type: RSZ, layer: 3, pos: 847
type: RSZ, layer: 3, pos: 94
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 985
type: RSZ, layer: 3, pos: 658
type: RSZ, layer: 3, pos: 362
type: RSZ, layer: 3, pos: 665
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 674
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 360
type: RSZ, layer: 3, pos: 702
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 54
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 982
type: RSZ, layer: 3, pos: 866
type: RSZ, layer: 3, pos: 852
type: RSZ, layer: 3, pos: 1013
type: RSZ, layer: 3, pos: 51
type: RSZ, layer: 3, pos: 989
type: RSZ, layer: 3, pos: 833
type: RSZ, layer: 3, pos: 664
type: RSZ, layer: 3, pos: 82
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 628
type: RSZ, layer: 3, pos: 694
type: RSZ, layer: 3, pos: 53
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 349
type: RSZ, layer: 3, pos: 258
type: RSZ, layer: 3, pos: 428
type: RSZ, layer: 3, pos: 333
type: RSZ, layer: 3, pos: 105
type: RSZ, layer: 3, pos: 971
type: RSZ, layer: 3, pos: 365
type: RSZ, layer: 3, pos: 202
type: RSZ, layer: 3, pos: 266
type: RSZ, layer: 3, pos: 651
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 595
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 681
type: RSZ, layer: 3, pos: 683
type: RSZ, layer: 3, pos: 645
type: RSZ, layer: 3, pos: 421
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 296
type: RSZ, layer: 3, pos: 845
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 280
type: RSZ, layer: 3, pos: 1006
type: RSZ, layer: 3, pos: 1007
type: RSZ, layer: 3, pos: 648
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 970
type: RSZ, layer: 3, pos: 260
type: RSZ, layer: 3, pos: 857
type: RSZ, layer: 3, pos: 341
type: RSZ, layer: 3, pos: 320
type: RSZ, layer: 3, pos: 55
type: RSZ, layer: 3, pos: 973
type: RSZ, layer: 3, pos: 301
type: RSZ, layer: 3, pos: 328
type: RSZ, layer: 3, pos: 623
type: RSZ, layer: 3, pos: 78
type: RSZ, layer: 3, pos: 650
type: RSZ, layer: 3, pos: 1011
type: RSZ, layer: 3, pos: 321
type: RSZ, layer: 3, pos: 598
type: RSZ, layer: 3, pos: 127
type: RSZ, layer: 3, pos: 207
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 597
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 649
type: RSZ, layer: 3, pos: 304
type: RSZ, layer: 3, pos: 596
type: RSZ, layer: 3, pos: 682
type: RSZ, layer: 3, pos: 701
type: RSZ, layer: 3, pos: 86
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 247
type: RSZ, layer: 3, pos: 109
type: RSZ, layer: 3, pos: 853
type: RSZ, layer: 3, pos: 1002
type: RSZ, layer: 3, pos: 261
type: RSZ, layer: 3, pos: 672
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 862
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 855
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 687
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 593
type: RSZ, layer: 3, pos: 243
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 205
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 1008
type: RSZ, layer: 3, pos: 630
type: RSZ, layer: 3, pos: 126
type: RSZ, layer: 3, pos: 602
type: RSZ, layer: 3, pos: 215
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 589
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 60
type: RSZ, layer: 3, pos: 367
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 835
type: RSZ, layer: 3, pos: 642
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 586
type: RSZ, layer: 3, pos: 609
type: RSZ, layer: 3, pos: 352
type: RSZ, layer: 3, pos: 263
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 696
type: RSZ, layer: 3, pos: 590
type: RSZ, layer: 3, pos: 1022
type: RSZ, layer: 3, pos: 587
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 288
type: RSZ, layer: 3, pos: 842
type: RSZ, layer: 3, pos: 327
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 836
type: RSZ, layer: 3, pos: 990
type: RSZ, layer: 3, pos: 269
type: RSZ, layer: 3, pos: 52
type: RSZ, layer: 3, pos: 981
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 264
type: RSZ, layer: 3, pos: 631
type: RSZ, layer: 3, pos: 636
type: RSZ, layer: 3, pos: 256
type: RSZ, layer: 3, pos: 966
type: RSZ, layer: 3, pos: 246
type: RSZ, layer: 3, pos: 873
type: RSZ, layer: 3, pos: 968
type: RSZ, layer: 3, pos: 594
type: RSZ, layer: 3, pos: 413
type: RSZ, layer: 3, pos: 622
type: RSZ, layer: 3, pos: 119
type: RSZ, layer: 3, pos: 1016
type: RSZ, layer: 3, pos: 653
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 639
type: RSZ, layer: 3, pos: 585
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 987
type: RSZ, layer: 3, pos: 606
type: RSZ, layer: 3, pos: 979
type: RSZ, layer: 3, pos: 405
type: RSZ, layer: 3, pos: 998
type: RSZ, layer: 3, pos: 641
type: RSZ, layer: 3, pos: 257
type: RSZ, layer: 3, pos: 267
type: RSZ, layer: 3, pos: 217
type: RSZ, layer: 3, pos: 680
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 618
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 617
type: RSZ, layer: 3, pos: 1001
type: RSZ, layer: 3, pos: 834
type: RSZ, layer: 3, pos: 97
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 685
type: RSZ, layer: 3, pos: 125
type: RSZ, layer: 3, pos: 983
type: RSZ, layer: 3, pos: 591
type: RSZ, layer: 3, pos: 368
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 87
type: RSZ, layer: 3, pos: 607
type: RSZ, layer: 3, pos: 88
type: RSZ, layer: 3, pos: 322
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 960
type: RSZ, layer: 3, pos: 691
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 861
type: RSZ, layer: 3, pos: 89
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 74
type: RSZ, layer: 3, pos: 629
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 262
type: RSZ, layer: 3, pos: 112
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 659
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 209
type: RSZ, layer: 3, pos: 601
type: RSZ, layer: 3, pos: 588
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 580
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 229

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1244699, upper bound: 14.5284629
time: 64.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.2150746, upper bound: 14.3862582
time: 61.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.5578117, -0.2815094, -37.5578117, -0.2815094, -37.2763023, 37.2763023
1: -17.6220856, 10.4812832, -17.6220856, 10.4812832, -28.1033688, 28.1033688
2: -14.4312439, 10.0883579, -14.4312439, 10.0883579, -24.5196018, 24.5196018
3: -14.9140787, 14.0766926, -14.9140787, 14.0766926, -28.9907722, 28.9907722
4: -15.1334782, 14.7552061, -15.1334782, 14.7552061, -29.8886833, 29.8886833
5: -14.2266178, 15.1941137, -14.2266178, 15.1941137, -29.4207306, 29.4207306
6: -20.8351784, 10.3227959, -20.8351784, 10.3227959, -31.1579742, 31.1579742
7: -17.3737793, 16.5286179, -17.3737793, 16.5286179, -33.5186157, 33.5169334
8: -16.3228550, 19.1706161, -16.3228550, 19.1706161, -35.4579506, 35.4577751
9: -15.1635761, 13.7782021, -15.1635761, 13.7782021, -28.7448692, 28.7427635
10: -23.5488243, 17.2546597, -23.5488243, 17.2546597, -40.8034821, 40.8034821
11: -26.2301464, 10.3602743, -26.2301464, 10.3602743, -36.5904198, 36.5904198
12: -24.2345219, 12.1104794, -24.2345219, 12.1104794, -36.3450012, 36.3450012
13: -22.1990852, 18.4545364, -22.1990852, 18.4545364, -40.6536217, 40.6536217
14: -47.8707161, -0.4218502, -47.8707161, -0.4218502, -47.2460098, 47.2699203
15: -19.6529541, 10.3302298, -19.6529541, 10.3302298, -29.9831848, 29.9831848
16: -24.9793587, 13.2617989, -24.9793587, 13.2617989, -37.7769585, 37.7731247
17: -43.9613876, 12.4236822, -43.9613876, 12.4236822, -55.0502319, 55.0716095
18: -20.4577484, 12.4874125, -20.4577484, 12.4874125, -32.9451599, 32.9451599
19: -17.9183464, 4.2855158, -17.9183464, 4.2855158, -22.2038612, 22.2038612
20: -15.2772102, 8.4781389, -15.2772102, 8.4781389, -23.7553482, 23.7553482
21: -25.8919182, 3.7857480, -25.8919182, 3.7857480, -29.6776657, 29.6776657
22: -32.9530487, -0.8713036, -32.9530487, -0.8713036, -30.7276688, 30.7307320
23: -17.9241028, 8.9438848, -17.9241028, 8.9438848, -26.8679886, 26.8679886
24: -25.2896576, 7.3427725, -25.2896576, 7.3427725, -31.1242485, 31.1306171
25: -18.3243713, 10.8286724, -18.3243713, 10.8286724, -29.1530437, 29.1530437
26: -23.7112923, 14.9050426, -23.7112923, 14.9050426, -38.6163330, 38.6163330
27: -26.2949791, 6.6883535, -26.2949791, 6.6883535, -31.9750328, 31.9650116
28: -17.3191872, 10.6421642, -17.3191872, 10.6421642, -27.7733154, 27.7734261
29: -40.1622734, -5.2830868, -40.1622734, -5.2830868, -33.9340591, 33.9386368
30: -20.8790817, 12.3533125, -20.8790817, 12.3533125, -33.2323952, 33.2323952
31: -23.7122688, 7.0026283, -23.7122688, 7.0026283, -30.7148972, 30.7148972
32: -27.6387272, 4.3835154, -27.6387272, 4.3835154, -31.1183624, 31.1142426
33: -30.5414295, 14.6008110, -30.5414295, 14.6008110, -44.2369614, 44.2362404
34: -25.9748363, 9.9279261, -25.9748363, 9.9279261, -35.9027634, 35.9027634
35: -27.7439537, 10.9692993, -27.7439537, 10.9692993, -38.2985458, 38.3000946
36: -27.1807117, 10.9039993, -27.1807117, 10.9039993, -37.6415329, 37.6395950
37: -37.2227592, 9.6494160, -37.2227592, 9.6494160, -45.5753174, 45.5734062
38: -29.7080078, 13.9970264, -29.7080078, 13.9970264, -43.7050323, 43.7050323
39: -38.4577255, 11.6211214, -38.4577255, 11.6211214, -49.4270325, 49.4276581
40: -30.4051437, 9.7967424, -30.4051437, 9.7967424, -38.5545731, 38.5549202
41: -22.3953133, 9.5435400, -22.3953133, 9.5435400, -31.9388542, 31.9361801
42: -16.3960018, 7.5732532, -16.3960018, 7.5732532, -23.6198311, 23.6143818

Time for backsubstitution: 0.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 229
type: RSZ, layer: 3, pos: 868
type: RSZ, layer: 3, pos: 292
type: RSZ, layer: 3, pos: 236
type: RSZ, layer: 3, pos: 228
type: RSZ, layer: 3, pos: 355
type: RSZ, layer: 3, pos: 357
type: RSZ, layer: 3, pos: 284
type: RSZ, layer: 3, pos: 997
type: RSZ, layer: 3, pos: 724
type: RSZ, layer: 3, pos: 380
type: RSZ, layer: 3, pos: 363
type: RSZ, layer: 3, pos: 892
type: RSZ, layer: 3, pos: 375
type: RSZ, layer: 3, pos: 353
type: RSZ, layer: 3, pos: 377
type: RSZ, layer: 3, pos: 293
type: RSZ, layer: 3, pos: 887
type: RSZ, layer: 3, pos: 348
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 869
type: RSZ, layer: 3, pos: 316
type: RSZ, layer: 3, pos: 875
type: RSZ, layer: 3, pos: 378
type: RSZ, layer: 3, pos: 289
type: RSZ, layer: 3, pos: 369
type: RSZ, layer: 3, pos: 893
type: RSZ, layer: 3, pos: 351
type: RSZ, layer: 3, pos: 991
type: RSZ, layer: 3, pos: 999
type: RSZ, layer: 3, pos: 988
type: RSZ, layer: 3, pos: 305
type: RSZ, layer: 3, pos: 871
type: RSZ, layer: 3, pos: 881
type: RSZ, layer: 3, pos: 996
type: RSZ, layer: 3, pos: 876
type: RSZ, layer: 3, pos: 383
type: RSZ, layer: 3, pos: 299
type: RSZ, layer: 3, pos: 361
type: RSZ, layer: 3, pos: 877
type: RSZ, layer: 3, pos: 1015
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 381
type: RSZ, layer: 3, pos: 291
type: RSZ, layer: 3, pos: 306
type: RSZ, layer: 3, pos: 993
type: RSZ, layer: 3, pos: 382
type: RSZ, layer: 3, pos: 851
type: RSZ, layer: 3, pos: 334
type: RSZ, layer: 3, pos: 843
type: RSZ, layer: 3, pos: 865
type: RSZ, layer: 3, pos: 314
type: RSZ, layer: 3, pos: 282
type: RSZ, layer: 3, pos: 858
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 859
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 889
type: RSZ, layer: 3, pos: 863
type: RSZ, layer: 3, pos: 895
type: RSZ, layer: 3, pos: 1009
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 274
type: RSZ, layer: 3, pos: 346
type: RSZ, layer: 3, pos: 283
type: RSZ, layer: 3, pos: 849
type: RSZ, layer: 3, pos: 1004
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 271
type: RSZ, layer: 3, pos: 379
type: RSZ, layer: 3, pos: 1023
type: RSZ, layer: 3, pos: 972
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 335
type: RSZ, layer: 3, pos: 850
type: RSZ, layer: 3, pos: 986
type: RSZ, layer: 3, pos: 279
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 700
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 980
type: RSZ, layer: 3, pos: 1005
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 884
type: RSZ, layer: 3, pos: 340
type: RSZ, layer: 3, pos: 1003
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 343
type: RSZ, layer: 3, pos: 846
type: RSZ, layer: 3, pos: 684
type: RSZ, layer: 3, pos: 689
type: RSZ, layer: 3, pos: 885
type: RSZ, layer: 3, pos: 315
type: RSZ, layer: 3, pos: 1020
type: RSZ, layer: 3, pos: 978
type: RSZ, layer: 3, pos: 995
type: RSZ, layer: 3, pos: 300
type: RSZ, layer: 3, pos: 331
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 319
type: RSZ, layer: 3, pos: 882
type: RSZ, layer: 3, pos: 364
type: RSZ, layer: 3, pos: 370
type: RSZ, layer: 3, pos: 84
type: RSZ, layer: 3, pos: 1021
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 673
type: RSZ, layer: 3, pos: 235
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 338
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 63
type: RSZ, layer: 3, pos: 699
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 58
type: RSZ, layer: 3, pos: 85
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 297
type: RSZ, layer: 3, pos: 239
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 339
type: RSZ, layer: 3, pos: 281
type: RSZ, layer: 3, pos: 69
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1019
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 344
type: RSZ, layer: 3, pos: 68
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 270
type: RSZ, layer: 3, pos: 883
type: RSZ, layer: 3, pos: 358
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 372
type: RSZ, layer: 3, pos: 666
type: RSZ, layer: 3, pos: 974
type: RSZ, layer: 3, pos: 667
type: RSZ, layer: 3, pos: 644
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 656
type: RSZ, layer: 3, pos: 870
type: RSZ, layer: 3, pos: 690
type: RSZ, layer: 3, pos: 273
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 336
type: RSZ, layer: 3, pos: 647
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 860
type: RSZ, layer: 3, pos: 841
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 318
type: RSZ, layer: 3, pos: 977
type: RSZ, layer: 3, pos: 61
type: RSZ, layer: 3, pos: 879
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 345
type: RSZ, layer: 3, pos: 663
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 110
type: RSZ, layer: 3, pos: 1017
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 102
type: RSZ, layer: 3, pos: 967
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 688
type: RSZ, layer: 3, pos: 50
type: RSZ, layer: 3, pos: 214
type: RSZ, layer: 3, pos: 867
type: RSZ, layer: 3, pos: 1014
type: RSZ, layer: 3, pos: 652
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 354
type: RSZ, layer: 3, pos: 114
type: RSZ, layer: 3, pos: 1012
type: RSZ, layer: 3, pos: 1018
type: RSZ, layer: 3, pos: 329
type: RSZ, layer: 3, pos: 844
type: RSZ, layer: 3, pos: 703
type: RSZ, layer: 3, pos: 326
type: RSZ, layer: 3, pos: 201
type: RSZ, layer: 3, pos: 376
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 1010
type: RSZ, layer: 3, pos: 259
type: RSZ, layer: 3, pos: 371
type: RSZ, layer: 3, pos: 874
type: RSZ, layer: 3, pos: 57
type: RSZ, layer: 3, pos: 272
type: RSZ, layer: 3, pos: 347
type: RSZ, layer: 3, pos: 692
type: RSZ, layer: 3, pos: 646
type: RSZ, layer: 3, pos: 275
type: RSZ, layer: 3, pos: 56
type: RSZ, layer: 3, pos: 203
type: RSZ, layer: 3, pos: 657
type: RSZ, layer: 3, pos: 420
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 276
type: RSZ, layer: 3, pos: 113
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 265
type: RSZ, layer: 3, pos: 419
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 210
type: RSZ, layer: 3, pos: 324
type: RSZ, layer: 3, pos: 123
type: RSZ, layer: 3, pos: 285
type: RSZ, layer: 3, pos: 965
type: RSZ, layer: 3, pos: 303
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 643
type: RSZ, layer: 3, pos: 847
type: RSZ, layer: 3, pos: 94
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 985
type: RSZ, layer: 3, pos: 658
type: RSZ, layer: 3, pos: 362
type: RSZ, layer: 3, pos: 665
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 674
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 360
type: RSZ, layer: 3, pos: 702
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 54
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 982
type: RSZ, layer: 3, pos: 866
type: RSZ, layer: 3, pos: 852
type: RSZ, layer: 3, pos: 1013
type: RSZ, layer: 3, pos: 51
type: RSZ, layer: 3, pos: 989
type: RSZ, layer: 3, pos: 833
type: RSZ, layer: 3, pos: 664
type: RSZ, layer: 3, pos: 82
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 628
type: RSZ, layer: 3, pos: 694
type: RSZ, layer: 3, pos: 53
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 349
type: RSZ, layer: 3, pos: 258
type: RSZ, layer: 3, pos: 428
type: RSZ, layer: 3, pos: 333
type: RSZ, layer: 3, pos: 105
type: RSZ, layer: 3, pos: 971
type: RSZ, layer: 3, pos: 365
type: RSZ, layer: 3, pos: 202
type: RSZ, layer: 3, pos: 266
type: RSZ, layer: 3, pos: 651
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 595
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 681
type: RSZ, layer: 3, pos: 683
type: RSZ, layer: 3, pos: 645
type: RSZ, layer: 3, pos: 421
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 296
type: RSZ, layer: 3, pos: 845
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 280
type: RSZ, layer: 3, pos: 1006
type: RSZ, layer: 3, pos: 1007
type: RSZ, layer: 3, pos: 648
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 970
type: RSZ, layer: 3, pos: 260
type: RSZ, layer: 3, pos: 857
type: RSZ, layer: 3, pos: 341
type: RSZ, layer: 3, pos: 320
type: RSZ, layer: 3, pos: 55
type: RSZ, layer: 3, pos: 973
type: RSZ, layer: 3, pos: 301
type: RSZ, layer: 3, pos: 328
type: RSZ, layer: 3, pos: 623
type: RSZ, layer: 3, pos: 78
type: RSZ, layer: 3, pos: 650
type: RSZ, layer: 3, pos: 1011
type: RSZ, layer: 3, pos: 321
type: RSZ, layer: 3, pos: 598
type: RSZ, layer: 3, pos: 127
type: RSZ, layer: 3, pos: 207
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 597
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 649
type: RSZ, layer: 3, pos: 304
type: RSZ, layer: 3, pos: 596
type: RSZ, layer: 3, pos: 682
type: RSZ, layer: 3, pos: 701
type: RSZ, layer: 3, pos: 86
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 247
type: RSZ, layer: 3, pos: 109
type: RSZ, layer: 3, pos: 853
type: RSZ, layer: 3, pos: 1002
type: RSZ, layer: 3, pos: 261
type: RSZ, layer: 3, pos: 672
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 862
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 855
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 687
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 593
type: RSZ, layer: 3, pos: 243
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 205
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 1008
type: RSZ, layer: 3, pos: 630
type: RSZ, layer: 3, pos: 126
type: RSZ, layer: 3, pos: 602
type: RSZ, layer: 3, pos: 215
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 589
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 60
type: RSZ, layer: 3, pos: 367
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 835
type: RSZ, layer: 3, pos: 642
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 586
type: RSZ, layer: 3, pos: 609
type: RSZ, layer: 3, pos: 352
type: RSZ, layer: 3, pos: 263
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 696
type: RSZ, layer: 3, pos: 590
type: RSZ, layer: 3, pos: 1022
type: RSZ, layer: 3, pos: 587
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 288
type: RSZ, layer: 3, pos: 842
type: RSZ, layer: 3, pos: 327
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 836
type: RSZ, layer: 3, pos: 990
type: RSZ, layer: 3, pos: 269
type: RSZ, layer: 3, pos: 52
type: RSZ, layer: 3, pos: 981
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 264
type: RSZ, layer: 3, pos: 631
type: RSZ, layer: 3, pos: 636
type: RSZ, layer: 3, pos: 256
type: RSZ, layer: 3, pos: 966
type: RSZ, layer: 3, pos: 246
type: RSZ, layer: 3, pos: 873
type: RSZ, layer: 3, pos: 968
type: RSZ, layer: 3, pos: 594
type: RSZ, layer: 3, pos: 413
type: RSZ, layer: 3, pos: 622
type: RSZ, layer: 3, pos: 119
type: RSZ, layer: 3, pos: 1016
type: RSZ, layer: 3, pos: 653
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 639
type: RSZ, layer: 3, pos: 585
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 987
type: RSZ, layer: 3, pos: 606
type: RSZ, layer: 3, pos: 979
type: RSZ, layer: 3, pos: 405
type: RSZ, layer: 3, pos: 998
type: RSZ, layer: 3, pos: 641
type: RSZ, layer: 3, pos: 257
type: RSZ, layer: 3, pos: 267
type: RSZ, layer: 3, pos: 217
type: RSZ, layer: 3, pos: 680
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 618
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 617
type: RSZ, layer: 3, pos: 1001
type: RSZ, layer: 3, pos: 834
type: RSZ, layer: 3, pos: 97
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 685
type: RSZ, layer: 3, pos: 125
type: RSZ, layer: 3, pos: 983
type: RSZ, layer: 3, pos: 591
type: RSZ, layer: 3, pos: 368
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 87
type: RSZ, layer: 3, pos: 607
type: RSZ, layer: 3, pos: 88
type: RSZ, layer: 3, pos: 322
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 960
type: RSZ, layer: 3, pos: 691
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 861
type: RSZ, layer: 3, pos: 89
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 74
type: RSZ, layer: 3, pos: 629
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 262
type: RSZ, layer: 3, pos: 112
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 659
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 209
type: RSZ, layer: 3, pos: 601
type: RSZ, layer: 3, pos: 588
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 580
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.3862582, upper bound: 14.2150746
time: 56.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.5284629, upper bound: 14.1244699
time: 59.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.5578117, -0.2815094, -37.5578117, -0.2815094, -37.2763023, 37.2763023
1: -17.6220856, 10.4812832, -17.6220856, 10.4812832, -28.1033688, 28.1033688
2: -14.4312439, 10.0883579, -14.4312439, 10.0883579, -24.5196018, 24.5196018
3: -14.9140787, 14.0766926, -14.9140787, 14.0766926, -28.9907722, 28.9907722
4: -15.1334782, 14.7552061, -15.1334782, 14.7552061, -29.8886833, 29.8886833
5: -14.2266178, 15.1941137, -14.2266178, 15.1941137, -29.4207306, 29.4207306
6: -20.8351784, 10.3227959, -20.8351784, 10.3227959, -31.1579742, 31.1579742
7: -17.3737793, 16.5286179, -17.3737793, 16.5286179, -33.5181122, 33.5176964
8: -16.3228550, 19.1706161, -16.3228550, 19.1706161, -35.4570961, 35.4586792
9: -15.1635761, 13.7782021, -15.1635761, 13.7782021, -28.7422447, 28.7455902
10: -23.5488243, 17.2546597, -23.5488243, 17.2546597, -40.8034821, 40.8034821
11: -26.2301464, 10.3602743, -26.2301464, 10.3602743, -36.5904198, 36.5904198
12: -24.2345219, 12.1104794, -24.2345219, 12.1104794, -36.3450012, 36.3450012
13: -22.1990852, 18.4545364, -22.1990852, 18.4545364, -40.6536217, 40.6536217
14: -47.8707161, -0.4218502, -47.8707161, -0.4218502, -47.2399139, 47.2761993
15: -19.6529541, 10.3302298, -19.6529541, 10.3302298, -29.9831848, 29.9831848
16: -24.9793587, 13.2617989, -24.9793587, 13.2617989, -37.7722435, 37.7778969
17: -43.9613876, 12.4236822, -43.9613876, 12.4236822, -55.0456085, 55.0764313
18: -20.4577484, 12.4874125, -20.4577484, 12.4874125, -32.9451599, 32.9451599
19: -17.9183464, 4.2855158, -17.9183464, 4.2855158, -22.2038612, 22.2038612
20: -15.2772102, 8.4781389, -15.2772102, 8.4781389, -23.7553482, 23.7553482
21: -25.8919182, 3.7857480, -25.8919182, 3.7857480, -29.6776657, 29.6776657
22: -32.9530487, -0.8713036, -32.9530487, -0.8713036, -30.7358856, 30.7225666
23: -17.9241028, 8.9438848, -17.9241028, 8.9438848, -26.8679886, 26.8679886
24: -25.2896576, 7.3427725, -25.2896576, 7.3427725, -31.1336708, 31.1214142
25: -18.3243713, 10.8286724, -18.3243713, 10.8286724, -29.1530437, 29.1530437
26: -23.7112923, 14.9050426, -23.7112923, 14.9050426, -38.6163330, 38.6163330
27: -26.2949791, 6.6883535, -26.2949791, 6.6883535, -31.9810066, 31.9595623
28: -17.3191872, 10.6421642, -17.3191872, 10.6421642, -27.7728348, 27.7739048
29: -40.1622734, -5.2830868, -40.1622734, -5.2830868, -33.9400558, 33.9326363
30: -20.8790817, 12.3533125, -20.8790817, 12.3533125, -33.2323952, 33.2323952
31: -23.7122688, 7.0026283, -23.7122688, 7.0026283, -30.7148972, 30.7148972
32: -27.6387272, 4.3835154, -27.6387272, 4.3835154, -31.1194153, 31.1131878
33: -30.5414295, 14.6008110, -30.5414295, 14.6008110, -44.2378769, 44.2355423
34: -25.9748363, 9.9279261, -25.9748363, 9.9279261, -35.9027634, 35.9027634
35: -27.7439537, 10.9692993, -27.7439537, 10.9692993, -38.2993240, 38.2997704
36: -27.1807117, 10.9039993, -27.1807117, 10.9039993, -37.6421814, 37.6389618
37: -37.2227592, 9.6494160, -37.2227592, 9.6494160, -45.5767517, 45.5719681
38: -29.7080078, 13.9970264, -29.7080078, 13.9970264, -43.7050323, 43.7050323
39: -38.4577255, 11.6211214, -38.4577255, 11.6211214, -49.4283905, 49.4263077
40: -30.4051437, 9.7967424, -30.4051437, 9.7967424, -38.5552292, 38.5542755
41: -22.3953133, 9.5435400, -22.3953133, 9.5435400, -31.9361420, 31.9388542
42: -16.3960018, 7.5732532, -16.3960018, 7.5732532, -23.6144981, 23.6215992

Time for backsubstitution: 0.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 229
type: RSZ, layer: 3, pos: 868
type: RSZ, layer: 3, pos: 292
type: RSZ, layer: 3, pos: 236
type: RSZ, layer: 3, pos: 228
type: RSZ, layer: 3, pos: 355
type: RSZ, layer: 3, pos: 357
type: RSZ, layer: 3, pos: 284
type: RSZ, layer: 3, pos: 997
type: RSZ, layer: 3, pos: 724
type: RSZ, layer: 3, pos: 380
type: RSZ, layer: 3, pos: 363
type: RSZ, layer: 3, pos: 892
type: RSZ, layer: 3, pos: 375
type: RSZ, layer: 3, pos: 353
type: RSZ, layer: 3, pos: 377
type: RSZ, layer: 3, pos: 293
type: RSZ, layer: 3, pos: 887
type: RSZ, layer: 3, pos: 348
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 869
type: RSZ, layer: 3, pos: 316
type: RSZ, layer: 3, pos: 875
type: RSZ, layer: 3, pos: 378
type: RSZ, layer: 3, pos: 289
type: RSZ, layer: 3, pos: 369
type: RSZ, layer: 3, pos: 893
type: RSZ, layer: 3, pos: 351
type: RSZ, layer: 3, pos: 991
type: RSZ, layer: 3, pos: 999
type: RSZ, layer: 3, pos: 988
type: RSZ, layer: 3, pos: 305
type: RSZ, layer: 3, pos: 871
type: RSZ, layer: 3, pos: 881
type: RSZ, layer: 3, pos: 996
type: RSZ, layer: 3, pos: 876
type: RSZ, layer: 3, pos: 383
type: RSZ, layer: 3, pos: 299
type: RSZ, layer: 3, pos: 361
type: RSZ, layer: 3, pos: 877
type: RSZ, layer: 3, pos: 1015
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 381
type: RSZ, layer: 3, pos: 291
type: RSZ, layer: 3, pos: 306
type: RSZ, layer: 3, pos: 993
type: RSZ, layer: 3, pos: 382
type: RSZ, layer: 3, pos: 851
type: RSZ, layer: 3, pos: 334
type: RSZ, layer: 3, pos: 843
type: RSZ, layer: 3, pos: 865
type: RSZ, layer: 3, pos: 314
type: RSZ, layer: 3, pos: 282
type: RSZ, layer: 3, pos: 858
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 859
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 889
type: RSZ, layer: 3, pos: 863
type: RSZ, layer: 3, pos: 895
type: RSZ, layer: 3, pos: 1009
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 274
type: RSZ, layer: 3, pos: 346
type: RSZ, layer: 3, pos: 283
type: RSZ, layer: 3, pos: 849
type: RSZ, layer: 3, pos: 1004
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 271
type: RSZ, layer: 3, pos: 379
type: RSZ, layer: 3, pos: 1023
type: RSZ, layer: 3, pos: 972
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 335
type: RSZ, layer: 3, pos: 850
type: RSZ, layer: 3, pos: 986
type: RSZ, layer: 3, pos: 279
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 700
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 980
type: RSZ, layer: 3, pos: 1005
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 884
type: RSZ, layer: 3, pos: 340
type: RSZ, layer: 3, pos: 1003
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 343
type: RSZ, layer: 3, pos: 846
type: RSZ, layer: 3, pos: 684
type: RSZ, layer: 3, pos: 689
type: RSZ, layer: 3, pos: 885
type: RSZ, layer: 3, pos: 315
type: RSZ, layer: 3, pos: 1020
type: RSZ, layer: 3, pos: 978
type: RSZ, layer: 3, pos: 995
type: RSZ, layer: 3, pos: 300
type: RSZ, layer: 3, pos: 331
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 319
type: RSZ, layer: 3, pos: 882
type: RSZ, layer: 3, pos: 364
type: RSZ, layer: 3, pos: 370
type: RSZ, layer: 3, pos: 84
type: RSZ, layer: 3, pos: 1021
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 673
type: RSZ, layer: 3, pos: 235
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 338
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 63
type: RSZ, layer: 3, pos: 699
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 58
type: RSZ, layer: 3, pos: 85
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 297
type: RSZ, layer: 3, pos: 239
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 339
type: RSZ, layer: 3, pos: 281
type: RSZ, layer: 3, pos: 69
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1019
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 344
type: RSZ, layer: 3, pos: 68
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 270
type: RSZ, layer: 3, pos: 883
type: RSZ, layer: 3, pos: 358
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 372
type: RSZ, layer: 3, pos: 666
type: RSZ, layer: 3, pos: 974
type: RSZ, layer: 3, pos: 667
type: RSZ, layer: 3, pos: 644
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 656
type: RSZ, layer: 3, pos: 870
type: RSZ, layer: 3, pos: 690
type: RSZ, layer: 3, pos: 273
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 336
type: RSZ, layer: 3, pos: 647
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 860
type: RSZ, layer: 3, pos: 841
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 318
type: RSZ, layer: 3, pos: 977
type: RSZ, layer: 3, pos: 61
type: RSZ, layer: 3, pos: 879
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 345
type: RSZ, layer: 3, pos: 663
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 110
type: RSZ, layer: 3, pos: 1017
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 102
type: RSZ, layer: 3, pos: 967
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 688
type: RSZ, layer: 3, pos: 50
type: RSZ, layer: 3, pos: 214
type: RSZ, layer: 3, pos: 867
type: RSZ, layer: 3, pos: 1014
type: RSZ, layer: 3, pos: 652
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 354
type: RSZ, layer: 3, pos: 114
type: RSZ, layer: 3, pos: 1012
type: RSZ, layer: 3, pos: 1018
type: RSZ, layer: 3, pos: 329
type: RSZ, layer: 3, pos: 844
type: RSZ, layer: 3, pos: 703
type: RSZ, layer: 3, pos: 326
type: RSZ, layer: 3, pos: 201
type: RSZ, layer: 3, pos: 376
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 1010
type: RSZ, layer: 3, pos: 259
type: RSZ, layer: 3, pos: 371
type: RSZ, layer: 3, pos: 874
type: RSZ, layer: 3, pos: 57
type: RSZ, layer: 3, pos: 272
type: RSZ, layer: 3, pos: 347
type: RSZ, layer: 3, pos: 692
type: RSZ, layer: 3, pos: 646
type: RSZ, layer: 3, pos: 275
type: RSZ, layer: 3, pos: 56
type: RSZ, layer: 3, pos: 203
type: RSZ, layer: 3, pos: 657
type: RSZ, layer: 3, pos: 420
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 276
type: RSZ, layer: 3, pos: 113
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 265
type: RSZ, layer: 3, pos: 419
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 210
type: RSZ, layer: 3, pos: 324
type: RSZ, layer: 3, pos: 123
type: RSZ, layer: 3, pos: 285
type: RSZ, layer: 3, pos: 965
type: RSZ, layer: 3, pos: 303
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 643
type: RSZ, layer: 3, pos: 847
type: RSZ, layer: 3, pos: 94
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 985
type: RSZ, layer: 3, pos: 658
type: RSZ, layer: 3, pos: 362
type: RSZ, layer: 3, pos: 665
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 674
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 360
type: RSZ, layer: 3, pos: 702
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 54
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 982
type: RSZ, layer: 3, pos: 866
type: RSZ, layer: 3, pos: 852
type: RSZ, layer: 3, pos: 1013
type: RSZ, layer: 3, pos: 51
type: RSZ, layer: 3, pos: 989
type: RSZ, layer: 3, pos: 833
type: RSZ, layer: 3, pos: 664
type: RSZ, layer: 3, pos: 82
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 628
type: RSZ, layer: 3, pos: 694
type: RSZ, layer: 3, pos: 53
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 349
type: RSZ, layer: 3, pos: 258
type: RSZ, layer: 3, pos: 428
type: RSZ, layer: 3, pos: 333
type: RSZ, layer: 3, pos: 105
type: RSZ, layer: 3, pos: 971
type: RSZ, layer: 3, pos: 365
type: RSZ, layer: 3, pos: 202
type: RSZ, layer: 3, pos: 266
type: RSZ, layer: 3, pos: 651
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 595
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 681
type: RSZ, layer: 3, pos: 683
type: RSZ, layer: 3, pos: 645
type: RSZ, layer: 3, pos: 421
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 296
type: RSZ, layer: 3, pos: 845
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 280
type: RSZ, layer: 3, pos: 1006
type: RSZ, layer: 3, pos: 1007
type: RSZ, layer: 3, pos: 648
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 970
type: RSZ, layer: 3, pos: 260
type: RSZ, layer: 3, pos: 857
type: RSZ, layer: 3, pos: 341
type: RSZ, layer: 3, pos: 320
type: RSZ, layer: 3, pos: 55
type: RSZ, layer: 3, pos: 973
type: RSZ, layer: 3, pos: 301
type: RSZ, layer: 3, pos: 328
type: RSZ, layer: 3, pos: 623
type: RSZ, layer: 3, pos: 78
type: RSZ, layer: 3, pos: 650
type: RSZ, layer: 3, pos: 1011
type: RSZ, layer: 3, pos: 321
type: RSZ, layer: 3, pos: 598
type: RSZ, layer: 3, pos: 127
type: RSZ, layer: 3, pos: 207
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 597
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 649
type: RSZ, layer: 3, pos: 304
type: RSZ, layer: 3, pos: 596
type: RSZ, layer: 3, pos: 682
type: RSZ, layer: 3, pos: 701
type: RSZ, layer: 3, pos: 86
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 247
type: RSZ, layer: 3, pos: 109
type: RSZ, layer: 3, pos: 853
type: RSZ, layer: 3, pos: 1002
type: RSZ, layer: 3, pos: 261
type: RSZ, layer: 3, pos: 672
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 862
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 855
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 687
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 593
type: RSZ, layer: 3, pos: 243
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 205
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 1008
type: RSZ, layer: 3, pos: 630
type: RSZ, layer: 3, pos: 126
type: RSZ, layer: 3, pos: 602
type: RSZ, layer: 3, pos: 215
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 589
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 60
type: RSZ, layer: 3, pos: 367
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 835
type: RSZ, layer: 3, pos: 642
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 586
type: RSZ, layer: 3, pos: 609
type: RSZ, layer: 3, pos: 352
type: RSZ, layer: 3, pos: 263
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 696
type: RSZ, layer: 3, pos: 590
type: RSZ, layer: 3, pos: 1022
type: RSZ, layer: 3, pos: 587
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 288
type: RSZ, layer: 3, pos: 842
type: RSZ, layer: 3, pos: 327
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 836
type: RSZ, layer: 3, pos: 990
type: RSZ, layer: 3, pos: 269
type: RSZ, layer: 3, pos: 52
type: RSZ, layer: 3, pos: 981
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 264
type: RSZ, layer: 3, pos: 631
type: RSZ, layer: 3, pos: 636
type: RSZ, layer: 3, pos: 256
type: RSZ, layer: 3, pos: 966
type: RSZ, layer: 3, pos: 246
type: RSZ, layer: 3, pos: 873
type: RSZ, layer: 3, pos: 968
type: RSZ, layer: 3, pos: 594
type: RSZ, layer: 3, pos: 413
type: RSZ, layer: 3, pos: 622
type: RSZ, layer: 3, pos: 119
type: RSZ, layer: 3, pos: 1016
type: RSZ, layer: 3, pos: 653
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 639
type: RSZ, layer: 3, pos: 585
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 987
type: RSZ, layer: 3, pos: 606
type: RSZ, layer: 3, pos: 979
type: RSZ, layer: 3, pos: 405
type: RSZ, layer: 3, pos: 998
type: RSZ, layer: 3, pos: 641
type: RSZ, layer: 3, pos: 257
type: RSZ, layer: 3, pos: 267
type: RSZ, layer: 3, pos: 217
type: RSZ, layer: 3, pos: 680
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 618
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 617
type: RSZ, layer: 3, pos: 1001
type: RSZ, layer: 3, pos: 834
type: RSZ, layer: 3, pos: 97
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 685
type: RSZ, layer: 3, pos: 125
type: RSZ, layer: 3, pos: 983
type: RSZ, layer: 3, pos: 591
type: RSZ, layer: 3, pos: 368
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 87
type: RSZ, layer: 3, pos: 607
type: RSZ, layer: 3, pos: 88
type: RSZ, layer: 3, pos: 322
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 960
type: RSZ, layer: 3, pos: 691
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 861
type: RSZ, layer: 3, pos: 89
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 74
type: RSZ, layer: 3, pos: 629
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 262
type: RSZ, layer: 3, pos: 112
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 659
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 209
type: RSZ, layer: 3, pos: 601
type: RSZ, layer: 3, pos: 588
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 580
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 229

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.5159429, upper bound: 14.1553625
time: 53.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.6363273, upper bound: 14.0397480
time: 55.25 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 109.50 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 109.50
Output dim: 4, lower bound: -14.0397480, upper bound: 14.6363273
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 109.50
Output dim: 4, lower bound: -14.1553625, upper bound: 14.5159429
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 109.50
Output dim: 4, lower bound: -14.1244699, upper bound: 14.5284629
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 109.50
Output dim: 4, lower bound: -14.2150746, upper bound: 14.3862582
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 109.50
Output dim: 4, lower bound: -14.3862582, upper bound: 14.2150746
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 109.50
Output dim: 4, lower bound: -14.5284629, upper bound: 14.1244699
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 109.50
Output dim: 4, lower bound: -14.5159429, upper bound: 14.1553625
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 109.50
Output dim: 4, lower bound: -14.6363273, upper bound: 14.0397480

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.5578117, -0.2815094, -37.5578117, -0.2815094, -37.2763023, 37.2763023
1: -17.6220856, 10.4812832, -17.6220856, 10.4812832, -28.1033688, 28.1033688
2: -14.4312439, 10.0883579, -14.4312439, 10.0883579, -24.5196018, 24.5196018
3: -14.9140787, 14.0766926, -14.9140787, 14.0766926, -28.9907722, 28.9907722
4: -15.1334782, 14.7552061, -15.1334782, 14.7552061, -29.8886833, 29.8886833
5: -14.2266178, 15.1941137, -14.2266178, 15.1941137, -29.4207306, 29.4207306
6: -20.8351784, 10.3227959, -20.8351784, 10.3227959, -31.1579742, 31.1579742
7: -17.3737793, 16.5286179, -17.3737793, 16.5286179, -33.4990616, 33.5000153
8: -16.3228550, 19.1706161, -16.3228550, 19.1706161, -35.4391174, 35.4393196
9: -15.1635761, 13.7782021, -15.1635761, 13.7782021, -28.7397575, 28.7375870
10: -23.5488243, 17.2546597, -23.5488243, 17.2546597, -40.8034821, 40.8034821
11: -26.2301464, 10.3602743, -26.2301464, 10.3602743, -36.5904198, 36.5904198
12: -24.2345219, 12.1104794, -24.2345219, 12.1104794, -36.3450012, 36.3450012
13: -22.1990852, 18.4545364, -22.1990852, 18.4545364, -40.6536217, 40.6536217
14: -47.8707161, -0.4218502, -47.8707161, -0.4218502, -47.2739334, 47.2372551
15: -19.6529541, 10.3302298, -19.6529541, 10.3302298, -29.9831848, 29.9831848
16: -24.9793587, 13.2617989, -24.9793587, 13.2617989, -37.7729187, 37.7722549
17: -43.9613876, 12.4236822, -43.9613876, 12.4236822, -55.0803833, 55.0487900
18: -20.4577484, 12.4874125, -20.4577484, 12.4874125, -32.9451599, 32.9451599
19: -17.9183464, 4.2855158, -17.9183464, 4.2855158, -22.2038612, 22.2038612
20: -15.2772102, 8.4781389, -15.2772102, 8.4781389, -23.7553482, 23.7553482
21: -25.8919182, 3.7857480, -25.8919182, 3.7857480, -29.6776657, 29.6776657
22: -32.9530487, -0.8713036, -32.9530487, -0.8713036, -30.6818619, 30.6855583
23: -17.9241028, 8.9438848, -17.9241028, 8.9438848, -26.8679886, 26.8679886
24: -25.2896576, 7.3427725, -25.2896576, 7.3427725, -31.1112137, 31.1177940
25: -18.3243713, 10.8286724, -18.3243713, 10.8286724, -29.1530437, 29.1530437
26: -23.7112923, 14.9050426, -23.7112923, 14.9050426, -38.6163330, 38.6163330
27: -26.2949791, 6.6883535, -26.2949791, 6.6883535, -31.9566650, 31.9774551
28: -17.3191872, 10.6421642, -17.3191872, 10.6421642, -27.7728043, 27.7714348
29: -40.1622734, -5.2830868, -40.1622734, -5.2830868, -33.8976555, 33.8952599
30: -20.8790817, 12.3533125, -20.8790817, 12.3533125, -33.2323952, 33.2323952
31: -23.7122688, 7.0026283, -23.7122688, 7.0026283, -30.7148972, 30.7148972
32: -27.6387272, 4.3835154, -27.6387272, 4.3835154, -31.0774765, 31.0839062
33: -30.5414295, 14.6008110, -30.5414295, 14.6008110, -44.2380600, 44.2438583
34: -25.9748363, 9.9279261, -25.9748363, 9.9279261, -35.9027634, 35.9027634
35: -27.7439537, 10.9692993, -27.7439537, 10.9692993, -38.3021507, 38.3045959
36: -27.1807117, 10.9039993, -27.1807117, 10.9039993, -37.6284637, 37.6318626
37: -37.2227592, 9.6494160, -37.2227592, 9.6494160, -45.5422058, 45.5484543
38: -29.7080078, 13.9970264, -29.7080078, 13.9970264, -43.7050323, 43.7050323
39: -38.4577255, 11.6211214, -38.4577255, 11.6211214, -49.4295578, 49.4305496
40: -30.4051437, 9.7967424, -30.4051437, 9.7967424, -38.5352020, 38.5366783
41: -22.3953133, 9.5435400, -22.3953133, 9.5435400, -31.9388542, 31.9369240
42: -16.3960018, 7.5732532, -16.3960018, 7.5732532, -23.6267300, 23.6210308

Time for backsubstitution: 0.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 868
type: RSZ, layer: 3, pos: 292
type: RSZ, layer: 3, pos: 236
type: RSZ, layer: 3, pos: 228
type: RSZ, layer: 3, pos: 355
type: RSZ, layer: 3, pos: 357
type: RSZ, layer: 3, pos: 284
type: RSZ, layer: 3, pos: 997
type: RSZ, layer: 3, pos: 724
type: RSZ, layer: 3, pos: 380
type: RSZ, layer: 3, pos: 363
type: RSZ, layer: 3, pos: 892
type: RSZ, layer: 3, pos: 375
type: RSZ, layer: 3, pos: 353
type: RSZ, layer: 3, pos: 377
type: RSZ, layer: 3, pos: 293
type: RSZ, layer: 3, pos: 887
type: RSZ, layer: 3, pos: 348
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 869
type: RSZ, layer: 3, pos: 316
type: RSZ, layer: 3, pos: 875
type: RSZ, layer: 3, pos: 378
type: RSZ, layer: 3, pos: 289
type: RSZ, layer: 3, pos: 369
type: RSZ, layer: 3, pos: 893
type: RSZ, layer: 3, pos: 351
type: RSZ, layer: 3, pos: 991
type: RSZ, layer: 3, pos: 999
type: RSZ, layer: 3, pos: 988
type: RSZ, layer: 3, pos: 305
type: RSZ, layer: 3, pos: 871
type: RSZ, layer: 3, pos: 881
type: RSZ, layer: 3, pos: 996
type: RSZ, layer: 3, pos: 876
type: RSZ, layer: 3, pos: 383
type: RSZ, layer: 3, pos: 299
type: RSZ, layer: 3, pos: 361
type: RSZ, layer: 3, pos: 877
type: RSZ, layer: 3, pos: 1015
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 381
type: RSZ, layer: 3, pos: 291
type: RSZ, layer: 3, pos: 306
type: RSZ, layer: 3, pos: 993
type: RSZ, layer: 3, pos: 382
type: RSZ, layer: 3, pos: 851
type: RSZ, layer: 3, pos: 334
type: RSZ, layer: 3, pos: 843
type: RSZ, layer: 3, pos: 865
type: RSZ, layer: 3, pos: 314
type: RSZ, layer: 3, pos: 282
type: RSZ, layer: 3, pos: 858
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 859
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 889
type: RSZ, layer: 3, pos: 863
type: RSZ, layer: 3, pos: 895
type: RSZ, layer: 3, pos: 1009
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 274
type: RSZ, layer: 3, pos: 346
type: RSZ, layer: 3, pos: 283
type: RSZ, layer: 3, pos: 849
type: RSZ, layer: 3, pos: 1004
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 271
type: RSZ, layer: 3, pos: 379
type: RSZ, layer: 3, pos: 1023
type: RSZ, layer: 3, pos: 972
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 335
type: RSZ, layer: 3, pos: 850
type: RSZ, layer: 3, pos: 986
type: RSZ, layer: 3, pos: 279
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 700
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 980
type: RSZ, layer: 3, pos: 1005
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 884
type: RSZ, layer: 3, pos: 340
type: RSZ, layer: 3, pos: 1003
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 343
type: RSZ, layer: 3, pos: 846
type: RSZ, layer: 3, pos: 684
type: RSZ, layer: 3, pos: 689
type: RSZ, layer: 3, pos: 885
type: RSZ, layer: 3, pos: 315
type: RSZ, layer: 3, pos: 1020
type: RSZ, layer: 3, pos: 978
type: RSZ, layer: 3, pos: 995
type: RSZ, layer: 3, pos: 300
type: RSZ, layer: 3, pos: 331
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 319
type: RSZ, layer: 3, pos: 882
type: RSZ, layer: 3, pos: 364
type: RSZ, layer: 3, pos: 370
type: RSZ, layer: 3, pos: 84
type: RSZ, layer: 3, pos: 1021
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 673
type: RSZ, layer: 3, pos: 235
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 338
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 63
type: RSZ, layer: 3, pos: 699
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 58
type: RSZ, layer: 3, pos: 85
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 297
type: RSZ, layer: 3, pos: 239
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 339
type: RSZ, layer: 3, pos: 281
type: RSZ, layer: 3, pos: 69
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1019
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 344
type: RSZ, layer: 3, pos: 68
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 270
type: RSZ, layer: 3, pos: 883
type: RSZ, layer: 3, pos: 358
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 372
type: RSZ, layer: 3, pos: 666
type: RSZ, layer: 3, pos: 974
type: RSZ, layer: 3, pos: 667
type: RSZ, layer: 3, pos: 644
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 656
type: RSZ, layer: 3, pos: 870
type: RSZ, layer: 3, pos: 690
type: RSZ, layer: 3, pos: 273
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 336
type: RSZ, layer: 3, pos: 647
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 860
type: RSZ, layer: 3, pos: 841
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 318
type: RSZ, layer: 3, pos: 977
type: RSZ, layer: 3, pos: 61
type: RSZ, layer: 3, pos: 879
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 345
type: RSZ, layer: 3, pos: 663
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 110
type: RSZ, layer: 3, pos: 1017
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 102
type: RSZ, layer: 3, pos: 967
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 688
type: RSZ, layer: 3, pos: 50
type: RSZ, layer: 3, pos: 214
type: RSZ, layer: 3, pos: 867
type: RSZ, layer: 3, pos: 1014
type: RSZ, layer: 3, pos: 652
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 354
type: RSZ, layer: 3, pos: 114
type: RSZ, layer: 3, pos: 1012
type: RSZ, layer: 3, pos: 1018
type: RSZ, layer: 3, pos: 329
type: RSZ, layer: 3, pos: 844
type: RSZ, layer: 3, pos: 703
type: RSZ, layer: 3, pos: 326
type: RSZ, layer: 3, pos: 201
type: RSZ, layer: 3, pos: 376
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 1010
type: RSZ, layer: 3, pos: 259
type: RSZ, layer: 3, pos: 371
type: RSZ, layer: 3, pos: 874
type: RSZ, layer: 3, pos: 57
type: RSZ, layer: 3, pos: 272
type: RSZ, layer: 3, pos: 347
type: RSZ, layer: 3, pos: 692
type: RSZ, layer: 3, pos: 646
type: RSZ, layer: 3, pos: 275
type: RSZ, layer: 3, pos: 56
type: RSZ, layer: 3, pos: 203
type: RSZ, layer: 3, pos: 657
type: RSZ, layer: 3, pos: 420
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 276
type: RSZ, layer: 3, pos: 113
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 265
type: RSZ, layer: 3, pos: 419
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 210
type: RSZ, layer: 3, pos: 324
type: RSZ, layer: 3, pos: 123
type: RSZ, layer: 3, pos: 285
type: RSZ, layer: 3, pos: 965
type: RSZ, layer: 3, pos: 303
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 643
type: RSZ, layer: 3, pos: 847
type: RSZ, layer: 3, pos: 94
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 985
type: RSZ, layer: 3, pos: 658
type: RSZ, layer: 3, pos: 362
type: RSZ, layer: 3, pos: 665
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 674
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 360
type: RSZ, layer: 3, pos: 702
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 54
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 982
type: RSZ, layer: 3, pos: 866
type: RSZ, layer: 3, pos: 852
type: RSZ, layer: 3, pos: 1013
type: RSZ, layer: 3, pos: 51
type: RSZ, layer: 3, pos: 989
type: RSZ, layer: 3, pos: 833
type: RSZ, layer: 3, pos: 664
type: RSZ, layer: 3, pos: 82
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 628
type: RSZ, layer: 3, pos: 694
type: RSZ, layer: 3, pos: 53
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 349
type: RSZ, layer: 3, pos: 258
type: RSZ, layer: 3, pos: 428
type: RSZ, layer: 3, pos: 333
type: RSZ, layer: 3, pos: 105
type: RSZ, layer: 3, pos: 971
type: RSZ, layer: 3, pos: 365
type: RSZ, layer: 3, pos: 202
type: RSZ, layer: 3, pos: 266
type: RSZ, layer: 3, pos: 651
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 595
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 681
type: RSZ, layer: 3, pos: 683
type: RSZ, layer: 3, pos: 645
type: RSZ, layer: 3, pos: 421
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 296
type: RSZ, layer: 3, pos: 845
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 280
type: RSZ, layer: 3, pos: 1006
type: RSZ, layer: 3, pos: 1007
type: RSZ, layer: 3, pos: 648
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 970
type: RSZ, layer: 3, pos: 260
type: RSZ, layer: 3, pos: 857
type: RSZ, layer: 3, pos: 341
type: RSZ, layer: 3, pos: 320
type: RSZ, layer: 3, pos: 55
type: RSZ, layer: 3, pos: 973
type: RSZ, layer: 3, pos: 301
type: RSZ, layer: 3, pos: 328
type: RSZ, layer: 3, pos: 623
type: RSZ, layer: 3, pos: 78
type: RSZ, layer: 3, pos: 650
type: RSZ, layer: 3, pos: 1011
type: RSZ, layer: 3, pos: 321
type: RSZ, layer: 3, pos: 598
type: RSZ, layer: 3, pos: 127
type: RSZ, layer: 3, pos: 207
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 597
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 649
type: RSZ, layer: 3, pos: 304
type: RSZ, layer: 3, pos: 596
type: RSZ, layer: 3, pos: 682
type: RSZ, layer: 3, pos: 701
type: RSZ, layer: 3, pos: 86
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 247
type: RSZ, layer: 3, pos: 109
type: RSZ, layer: 3, pos: 853
type: RSZ, layer: 3, pos: 1002
type: RSZ, layer: 3, pos: 261
type: RSZ, layer: 3, pos: 672
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 862
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 855
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 687
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 593
type: RSZ, layer: 3, pos: 243
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 205
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 1008
type: RSZ, layer: 3, pos: 630
type: RSZ, layer: 3, pos: 126
type: RSZ, layer: 3, pos: 602
type: RSZ, layer: 3, pos: 215
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 589
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 60
type: RSZ, layer: 3, pos: 367
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 835
type: RSZ, layer: 3, pos: 642
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 586
type: RSZ, layer: 3, pos: 609
type: RSZ, layer: 3, pos: 352
type: RSZ, layer: 3, pos: 263
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 696
type: RSZ, layer: 3, pos: 590
type: RSZ, layer: 3, pos: 1022
type: RSZ, layer: 3, pos: 587
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 288
type: RSZ, layer: 3, pos: 842
type: RSZ, layer: 3, pos: 327
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 836
type: RSZ, layer: 3, pos: 990
type: RSZ, layer: 3, pos: 269
type: RSZ, layer: 3, pos: 52
type: RSZ, layer: 3, pos: 981
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 264
type: RSZ, layer: 3, pos: 631
type: RSZ, layer: 3, pos: 636
type: RSZ, layer: 3, pos: 256
type: RSZ, layer: 3, pos: 966
type: RSZ, layer: 3, pos: 246
type: RSZ, layer: 3, pos: 873
type: RSZ, layer: 3, pos: 968
type: RSZ, layer: 3, pos: 594
type: RSZ, layer: 3, pos: 413
type: RSZ, layer: 3, pos: 622
type: RSZ, layer: 3, pos: 119
type: RSZ, layer: 3, pos: 1016
type: RSZ, layer: 3, pos: 653
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 639
type: RSZ, layer: 3, pos: 585
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 987
type: RSZ, layer: 3, pos: 606
type: RSZ, layer: 3, pos: 979
type: RSZ, layer: 3, pos: 405
type: RSZ, layer: 3, pos: 998
type: RSZ, layer: 3, pos: 641
type: RSZ, layer: 3, pos: 257
type: RSZ, layer: 3, pos: 267
type: RSZ, layer: 3, pos: 217
type: RSZ, layer: 3, pos: 680
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 618
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 617
type: RSZ, layer: 3, pos: 1001
type: RSZ, layer: 3, pos: 834
type: RSZ, layer: 3, pos: 97
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 685
type: RSZ, layer: 3, pos: 125
type: RSZ, layer: 3, pos: 983
type: RSZ, layer: 3, pos: 591
type: RSZ, layer: 3, pos: 368
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 87
type: RSZ, layer: 3, pos: 607
type: RSZ, layer: 3, pos: 88
type: RSZ, layer: 3, pos: 322
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 960
type: RSZ, layer: 3, pos: 691
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 861
type: RSZ, layer: 3, pos: 89
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 74
type: RSZ, layer: 3, pos: 629
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 262
type: RSZ, layer: 3, pos: 112
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 659
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 209
type: RSZ, layer: 3, pos: 601
type: RSZ, layer: 3, pos: 588
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 580
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 868

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -13.9782079, upper bound: 14.6311161
time: 43.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -13.9933462, upper bound: 14.1828287
time: 68.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.5578117, -0.2815094, -37.5578117, -0.2815094, -37.2763023, 37.2763023
1: -17.6220856, 10.4812832, -17.6220856, 10.4812832, -28.1033688, 28.1033688
2: -14.4312439, 10.0883579, -14.4312439, 10.0883579, -24.5196018, 24.5196018
3: -14.9140787, 14.0766926, -14.9140787, 14.0766926, -28.9907722, 28.9907722
4: -15.1334782, 14.7552061, -15.1334782, 14.7552061, -29.8886833, 29.8886833
5: -14.2266178, 15.1941137, -14.2266178, 15.1941137, -29.4207306, 29.4207306
6: -20.8351784, 10.3227959, -20.8351784, 10.3227959, -31.1579742, 31.1579742
7: -17.3737793, 16.5286179, -17.3737793, 16.5286179, -33.4994736, 33.4994774
8: -16.3228550, 19.1706161, -16.3228550, 19.1706161, -35.4410095, 35.4375305
9: -15.1635761, 13.7782021, -15.1635761, 13.7782021, -28.7408867, 28.7364120
10: -23.5488243, 17.2546597, -23.5488243, 17.2546597, -40.8034821, 40.8034821
11: -26.2301464, 10.3602743, -26.2301464, 10.3602743, -36.5904198, 36.5904198
12: -24.2345219, 12.1104794, -24.2345219, 12.1104794, -36.3450012, 36.3450012
13: -22.1990852, 18.4545364, -22.1990852, 18.4545364, -40.6536217, 40.6536217
14: -47.8707161, -0.4218502, -47.8707161, -0.4218502, -47.2735443, 47.2374077
15: -19.6529541, 10.3302298, -19.6529541, 10.3302298, -29.9831848, 29.9831848
16: -24.9793587, 13.2617989, -24.9793587, 13.2617989, -37.7779312, 37.7672577
17: -43.9613876, 12.4236822, -43.9613876, 12.4236822, -55.0794373, 55.0495453
18: -20.4577484, 12.4874125, -20.4577484, 12.4874125, -32.9451599, 32.9451599
19: -17.9183464, 4.2855158, -17.9183464, 4.2855158, -22.2038612, 22.2038612
20: -15.2772102, 8.4781389, -15.2772102, 8.4781389, -23.7553482, 23.7553482
21: -25.8919182, 3.7857480, -25.8919182, 3.7857480, -29.6776657, 29.6776657
22: -32.9530487, -0.8713036, -32.9530487, -0.8713036, -30.6722412, 30.6945591
23: -17.9241028, 8.9438848, -17.9241028, 8.9438848, -26.8679886, 26.8679886
24: -25.2896576, 7.3427725, -25.2896576, 7.3427725, -31.1055374, 31.1230068
25: -18.3243713, 10.8286724, -18.3243713, 10.8286724, -29.1530437, 29.1530437
26: -23.7112923, 14.9050426, -23.7112923, 14.9050426, -38.6163330, 38.6163330
27: -26.2949791, 6.6883535, -26.2949791, 6.6883535, -31.9560089, 31.9780655
28: -17.3191872, 10.6421642, -17.3191872, 10.6421642, -27.7724991, 27.7717304
29: -40.1622734, -5.2830868, -40.1622734, -5.2830868, -33.8878365, 33.9048119
30: -20.8790817, 12.3533125, -20.8790817, 12.3533125, -33.2323952, 33.2323952
31: -23.7122688, 7.0026283, -23.7122688, 7.0026283, -30.7148972, 30.7148972
32: -27.6387272, 4.3835154, -27.6387272, 4.3835154, -31.0776749, 31.0836277
33: -30.5414295, 14.6008110, -30.5414295, 14.6008110, -44.2415161, 44.2403412
34: -25.9748363, 9.9279261, -25.9748363, 9.9279261, -35.9027634, 35.9027634
35: -27.7439537, 10.9692993, -27.7439537, 10.9692993, -38.3050423, 38.3017349
36: -27.1807117, 10.9039993, -27.1807117, 10.9039993, -37.6286316, 37.6316566
37: -37.2227592, 9.6494160, -37.2227592, 9.6494160, -45.5436707, 45.5469742
38: -29.7080078, 13.9970264, -29.7080078, 13.9970264, -43.7050323, 43.7050323
39: -38.4577255, 11.6211214, -38.4577255, 11.6211214, -49.4284668, 49.4315796
40: -30.4051437, 9.7967424, -30.4051437, 9.7967424, -38.5357208, 38.5361366
41: -22.3953133, 9.5435400, -22.3953133, 9.5435400, -31.9388542, 31.9364815
42: -16.3960018, 7.5732532, -16.3960018, 7.5732532, -23.6281338, 23.6196289

Time for backsubstitution: 0.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 868
type: RSZ, layer: 3, pos: 292
type: RSZ, layer: 3, pos: 236
type: RSZ, layer: 3, pos: 228
type: RSZ, layer: 3, pos: 355
type: RSZ, layer: 3, pos: 357
type: RSZ, layer: 3, pos: 284
type: RSZ, layer: 3, pos: 997
type: RSZ, layer: 3, pos: 724
type: RSZ, layer: 3, pos: 380
type: RSZ, layer: 3, pos: 363
type: RSZ, layer: 3, pos: 892
type: RSZ, layer: 3, pos: 375
type: RSZ, layer: 3, pos: 353
type: RSZ, layer: 3, pos: 377
type: RSZ, layer: 3, pos: 293
type: RSZ, layer: 3, pos: 887
type: RSZ, layer: 3, pos: 348
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 869
type: RSZ, layer: 3, pos: 316
type: RSZ, layer: 3, pos: 875
type: RSZ, layer: 3, pos: 378
type: RSZ, layer: 3, pos: 289
type: RSZ, layer: 3, pos: 369
type: RSZ, layer: 3, pos: 893
type: RSZ, layer: 3, pos: 351
type: RSZ, layer: 3, pos: 991
type: RSZ, layer: 3, pos: 999
type: RSZ, layer: 3, pos: 988
type: RSZ, layer: 3, pos: 305
type: RSZ, layer: 3, pos: 871
type: RSZ, layer: 3, pos: 881
type: RSZ, layer: 3, pos: 996
type: RSZ, layer: 3, pos: 876
type: RSZ, layer: 3, pos: 383
type: RSZ, layer: 3, pos: 299
type: RSZ, layer: 3, pos: 361
type: RSZ, layer: 3, pos: 877
type: RSZ, layer: 3, pos: 1015
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 381
type: RSZ, layer: 3, pos: 291
type: RSZ, layer: 3, pos: 306
type: RSZ, layer: 3, pos: 993
type: RSZ, layer: 3, pos: 382
type: RSZ, layer: 3, pos: 851
type: RSZ, layer: 3, pos: 334
type: RSZ, layer: 3, pos: 843
type: RSZ, layer: 3, pos: 865
type: RSZ, layer: 3, pos: 314
type: RSZ, layer: 3, pos: 282
type: RSZ, layer: 3, pos: 858
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 859
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 889
type: RSZ, layer: 3, pos: 863
type: RSZ, layer: 3, pos: 895
type: RSZ, layer: 3, pos: 1009
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 274
type: RSZ, layer: 3, pos: 346
type: RSZ, layer: 3, pos: 283
type: RSZ, layer: 3, pos: 849
type: RSZ, layer: 3, pos: 1004
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 271
type: RSZ, layer: 3, pos: 379
type: RSZ, layer: 3, pos: 1023
type: RSZ, layer: 3, pos: 972
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 335
type: RSZ, layer: 3, pos: 850
type: RSZ, layer: 3, pos: 986
type: RSZ, layer: 3, pos: 279
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 700
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 980
type: RSZ, layer: 3, pos: 1005
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 884
type: RSZ, layer: 3, pos: 340
type: RSZ, layer: 3, pos: 1003
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 343
type: RSZ, layer: 3, pos: 846
type: RSZ, layer: 3, pos: 684
type: RSZ, layer: 3, pos: 689
type: RSZ, layer: 3, pos: 885
type: RSZ, layer: 3, pos: 315
type: RSZ, layer: 3, pos: 1020
type: RSZ, layer: 3, pos: 978
type: RSZ, layer: 3, pos: 995
type: RSZ, layer: 3, pos: 300
type: RSZ, layer: 3, pos: 331
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 319
type: RSZ, layer: 3, pos: 882
type: RSZ, layer: 3, pos: 364
type: RSZ, layer: 3, pos: 370
type: RSZ, layer: 3, pos: 84
type: RSZ, layer: 3, pos: 1021
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 673
type: RSZ, layer: 3, pos: 235
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 338
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 63
type: RSZ, layer: 3, pos: 699
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 58
type: RSZ, layer: 3, pos: 85
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 297
type: RSZ, layer: 3, pos: 239
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 339
type: RSZ, layer: 3, pos: 281
type: RSZ, layer: 3, pos: 69
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1019
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 344
type: RSZ, layer: 3, pos: 68
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 270
type: RSZ, layer: 3, pos: 883
type: RSZ, layer: 3, pos: 358
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 372
type: RSZ, layer: 3, pos: 666
type: RSZ, layer: 3, pos: 974
type: RSZ, layer: 3, pos: 667
type: RSZ, layer: 3, pos: 644
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 656
type: RSZ, layer: 3, pos: 870
type: RSZ, layer: 3, pos: 690
type: RSZ, layer: 3, pos: 273
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 336
type: RSZ, layer: 3, pos: 647
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 860
type: RSZ, layer: 3, pos: 841
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 318
type: RSZ, layer: 3, pos: 977
type: RSZ, layer: 3, pos: 61
type: RSZ, layer: 3, pos: 879
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 345
type: RSZ, layer: 3, pos: 663
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 110
type: RSZ, layer: 3, pos: 1017
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 102
type: RSZ, layer: 3, pos: 967
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 688
type: RSZ, layer: 3, pos: 50
type: RSZ, layer: 3, pos: 214
type: RSZ, layer: 3, pos: 867
type: RSZ, layer: 3, pos: 1014
type: RSZ, layer: 3, pos: 652
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 354
type: RSZ, layer: 3, pos: 114
type: RSZ, layer: 3, pos: 1012
type: RSZ, layer: 3, pos: 1018
type: RSZ, layer: 3, pos: 329
type: RSZ, layer: 3, pos: 844
type: RSZ, layer: 3, pos: 703
type: RSZ, layer: 3, pos: 326
type: RSZ, layer: 3, pos: 201
type: RSZ, layer: 3, pos: 376
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 1010
type: RSZ, layer: 3, pos: 259
type: RSZ, layer: 3, pos: 371
type: RSZ, layer: 3, pos: 874
type: RSZ, layer: 3, pos: 57
type: RSZ, layer: 3, pos: 272
type: RSZ, layer: 3, pos: 347
type: RSZ, layer: 3, pos: 692
type: RSZ, layer: 3, pos: 646
type: RSZ, layer: 3, pos: 275
type: RSZ, layer: 3, pos: 56
type: RSZ, layer: 3, pos: 203
type: RSZ, layer: 3, pos: 657
type: RSZ, layer: 3, pos: 420
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 276
type: RSZ, layer: 3, pos: 113
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 265
type: RSZ, layer: 3, pos: 419
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 210
type: RSZ, layer: 3, pos: 324
type: RSZ, layer: 3, pos: 123
type: RSZ, layer: 3, pos: 285
type: RSZ, layer: 3, pos: 965
type: RSZ, layer: 3, pos: 303
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 643
type: RSZ, layer: 3, pos: 847
type: RSZ, layer: 3, pos: 94
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 985
type: RSZ, layer: 3, pos: 658
type: RSZ, layer: 3, pos: 362
type: RSZ, layer: 3, pos: 665
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 674
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 360
type: RSZ, layer: 3, pos: 702
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 54
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 982
type: RSZ, layer: 3, pos: 866
type: RSZ, layer: 3, pos: 852
type: RSZ, layer: 3, pos: 1013
type: RSZ, layer: 3, pos: 51
type: RSZ, layer: 3, pos: 989
type: RSZ, layer: 3, pos: 833
type: RSZ, layer: 3, pos: 664
type: RSZ, layer: 3, pos: 82
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 628
type: RSZ, layer: 3, pos: 694
type: RSZ, layer: 3, pos: 53
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 349
type: RSZ, layer: 3, pos: 258
type: RSZ, layer: 3, pos: 428
type: RSZ, layer: 3, pos: 333
type: RSZ, layer: 3, pos: 105
type: RSZ, layer: 3, pos: 971
type: RSZ, layer: 3, pos: 365
type: RSZ, layer: 3, pos: 202
type: RSZ, layer: 3, pos: 266
type: RSZ, layer: 3, pos: 651
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 595
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 681
type: RSZ, layer: 3, pos: 683
type: RSZ, layer: 3, pos: 645
type: RSZ, layer: 3, pos: 421
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 296
type: RSZ, layer: 3, pos: 845
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 280
type: RSZ, layer: 3, pos: 1006
type: RSZ, layer: 3, pos: 1007
type: RSZ, layer: 3, pos: 648
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 970
type: RSZ, layer: 3, pos: 260
type: RSZ, layer: 3, pos: 857
type: RSZ, layer: 3, pos: 341
type: RSZ, layer: 3, pos: 320
type: RSZ, layer: 3, pos: 55
type: RSZ, layer: 3, pos: 973
type: RSZ, layer: 3, pos: 301
type: RSZ, layer: 3, pos: 328
type: RSZ, layer: 3, pos: 623
type: RSZ, layer: 3, pos: 78
type: RSZ, layer: 3, pos: 650
type: RSZ, layer: 3, pos: 1011
type: RSZ, layer: 3, pos: 321
type: RSZ, layer: 3, pos: 598
type: RSZ, layer: 3, pos: 127
type: RSZ, layer: 3, pos: 207
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 597
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 649
type: RSZ, layer: 3, pos: 304
type: RSZ, layer: 3, pos: 596
type: RSZ, layer: 3, pos: 682
type: RSZ, layer: 3, pos: 701
type: RSZ, layer: 3, pos: 86
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 247
type: RSZ, layer: 3, pos: 109
type: RSZ, layer: 3, pos: 853
type: RSZ, layer: 3, pos: 1002
type: RSZ, layer: 3, pos: 261
type: RSZ, layer: 3, pos: 672
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 862
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 855
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 687
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 593
type: RSZ, layer: 3, pos: 243
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 205
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 1008
type: RSZ, layer: 3, pos: 630
type: RSZ, layer: 3, pos: 126
type: RSZ, layer: 3, pos: 602
type: RSZ, layer: 3, pos: 215
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 589
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 60
type: RSZ, layer: 3, pos: 367
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 835
type: RSZ, layer: 3, pos: 642
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 586
type: RSZ, layer: 3, pos: 609
type: RSZ, layer: 3, pos: 352
type: RSZ, layer: 3, pos: 263
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 696
type: RSZ, layer: 3, pos: 590
type: RSZ, layer: 3, pos: 1022
type: RSZ, layer: 3, pos: 587
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 288
type: RSZ, layer: 3, pos: 842
type: RSZ, layer: 3, pos: 327
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 836
type: RSZ, layer: 3, pos: 990
type: RSZ, layer: 3, pos: 269
type: RSZ, layer: 3, pos: 52
type: RSZ, layer: 3, pos: 981
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 264
type: RSZ, layer: 3, pos: 631
type: RSZ, layer: 3, pos: 636
type: RSZ, layer: 3, pos: 256
type: RSZ, layer: 3, pos: 966
type: RSZ, layer: 3, pos: 246
type: RSZ, layer: 3, pos: 873
type: RSZ, layer: 3, pos: 968
type: RSZ, layer: 3, pos: 594
type: RSZ, layer: 3, pos: 413
type: RSZ, layer: 3, pos: 622
type: RSZ, layer: 3, pos: 119
type: RSZ, layer: 3, pos: 1016
type: RSZ, layer: 3, pos: 653
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 639
type: RSZ, layer: 3, pos: 585
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 987
type: RSZ, layer: 3, pos: 606
type: RSZ, layer: 3, pos: 979
type: RSZ, layer: 3, pos: 405
type: RSZ, layer: 3, pos: 998
type: RSZ, layer: 3, pos: 641
type: RSZ, layer: 3, pos: 257
type: RSZ, layer: 3, pos: 267
type: RSZ, layer: 3, pos: 217
type: RSZ, layer: 3, pos: 680
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 618
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 617
type: RSZ, layer: 3, pos: 1001
type: RSZ, layer: 3, pos: 834
type: RSZ, layer: 3, pos: 97
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 685
type: RSZ, layer: 3, pos: 125
type: RSZ, layer: 3, pos: 983
type: RSZ, layer: 3, pos: 591
type: RSZ, layer: 3, pos: 368
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 87
type: RSZ, layer: 3, pos: 607
type: RSZ, layer: 3, pos: 88
type: RSZ, layer: 3, pos: 322
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 960
type: RSZ, layer: 3, pos: 691
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 861
type: RSZ, layer: 3, pos: 89
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 74
type: RSZ, layer: 3, pos: 629
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 262
type: RSZ, layer: 3, pos: 112
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 659
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 209
type: RSZ, layer: 3, pos: 601
type: RSZ, layer: 3, pos: 588
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 580
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 868

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.0843435, upper bound: 14.5099357
time: 53.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1091648, upper bound: 14.1064588
time: 62.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.5578117, -0.2815094, -37.5578117, -0.2815094, -37.2763023, 37.2763023
1: -17.6220856, 10.4812832, -17.6220856, 10.4812832, -28.1033688, 28.1033688
2: -14.4312439, 10.0883579, -14.4312439, 10.0883579, -24.5196018, 24.5196018
3: -14.9140787, 14.0766926, -14.9140787, 14.0766926, -28.9907722, 28.9907722
4: -15.1334782, 14.7552061, -15.1334782, 14.7552061, -29.8886833, 29.8886833
5: -14.2266178, 15.1941137, -14.2266178, 15.1941137, -29.4207306, 29.4207306
6: -20.8351784, 10.3227959, -20.8351784, 10.3227959, -31.1579742, 31.1579742
7: -17.3737793, 16.5286179, -17.3737793, 16.5286179, -33.4982986, 33.5006256
8: -16.3228550, 19.1706161, -16.3228550, 19.1706161, -35.4382172, 35.4403687
9: -15.1635761, 13.7782021, -15.1635761, 13.7782021, -28.7369270, 28.7402172
10: -23.5488243, 17.2546597, -23.5488243, 17.2546597, -40.8034821, 40.8034821
11: -26.2301464, 10.3602743, -26.2301464, 10.3602743, -36.5904198, 36.5904198
12: -24.2345219, 12.1104794, -24.2345219, 12.1104794, -36.3450012, 36.3450012
13: -22.1990852, 18.4545364, -22.1990852, 18.4545364, -40.6536217, 40.6536217
14: -47.8707161, -0.4218502, -47.8707161, -0.4218502, -47.2676544, 47.2433472
15: -19.6529541, 10.3302298, -19.6529541, 10.3302298, -29.9831848, 29.9831848
16: -24.9793587, 13.2617989, -24.9793587, 13.2617989, -37.7681427, 37.7770462
17: -43.9613876, 12.4236822, -43.9613876, 12.4236822, -55.0755615, 55.0533447
18: -20.4577484, 12.4874125, -20.4577484, 12.4874125, -32.9451599, 32.9451599
19: -17.9183464, 4.2855158, -17.9183464, 4.2855158, -22.2038612, 22.2038612
20: -15.2772102, 8.4781389, -15.2772102, 8.4781389, -23.7553482, 23.7553482
21: -25.8919182, 3.7857480, -25.8919182, 3.7857480, -29.6776657, 29.6776657
22: -32.9530487, -0.8713036, -32.9530487, -0.8713036, -30.6899643, 30.6773415
23: -17.9241028, 8.9438848, -17.9241028, 8.9438848, -26.8679886, 26.8679886
24: -25.2896576, 7.3427725, -25.2896576, 7.3427725, -31.1202850, 31.1083717
25: -18.3243713, 10.8286724, -18.3243713, 10.8286724, -29.1530437, 29.1530437
26: -23.7112923, 14.9050426, -23.7112923, 14.9050426, -38.6163330, 38.6163330
27: -26.2949791, 6.6883535, -26.2949791, 6.6883535, -31.9622192, 31.9714813
28: -17.3191872, 10.6421642, -17.3191872, 10.6421642, -27.7723236, 27.7719116
29: -40.1622734, -5.2830868, -40.1622734, -5.2830868, -33.9035835, 33.8892593
30: -20.8790817, 12.3533125, -20.8790817, 12.3533125, -33.2323952, 33.2323952
31: -23.7122688, 7.0026283, -23.7122688, 7.0026283, -30.7148972, 30.7148972
32: -27.6387272, 4.3835154, -27.6387272, 4.3835154, -31.0784988, 31.0828514
33: -30.5414295, 14.6008110, -30.5414295, 14.6008110, -44.2389832, 44.2429352
34: -25.9748363, 9.9279261, -25.9748363, 9.9279261, -35.9027634, 35.9027634
35: -27.7439537, 10.9692993, -27.7439537, 10.9692993, -38.3028526, 38.3038139
36: -27.1807117, 10.9039993, -27.1807117, 10.9039993, -37.6290894, 37.6312141
37: -37.2227592, 9.6494160, -37.2227592, 9.6494160, -45.5436401, 45.5470161
38: -29.7080078, 13.9970264, -29.7080078, 13.9970264, -43.7050323, 43.7050323
39: -38.4577255, 11.6211214, -38.4577255, 11.6211214, -49.4308701, 49.4291992
40: -30.4051437, 9.7967424, -30.4051437, 9.7967424, -38.5358582, 38.5360336
41: -22.3953133, 9.5435400, -22.3953133, 9.5435400, -31.9365082, 31.9388542
42: -16.3960018, 7.5732532, -16.3960018, 7.5732532, -23.6195126, 23.6266403

Time for backsubstitution: 0.92 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 868
type: RSZ, layer: 3, pos: 292
type: RSZ, layer: 3, pos: 236
type: RSZ, layer: 3, pos: 228
type: RSZ, layer: 3, pos: 355
type: RSZ, layer: 3, pos: 357
type: RSZ, layer: 3, pos: 284
type: RSZ, layer: 3, pos: 997
type: RSZ, layer: 3, pos: 724
type: RSZ, layer: 3, pos: 380
type: RSZ, layer: 3, pos: 363
type: RSZ, layer: 3, pos: 892
type: RSZ, layer: 3, pos: 375
type: RSZ, layer: 3, pos: 353
type: RSZ, layer: 3, pos: 377
type: RSZ, layer: 3, pos: 293
type: RSZ, layer: 3, pos: 887
type: RSZ, layer: 3, pos: 348
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 869
type: RSZ, layer: 3, pos: 316
type: RSZ, layer: 3, pos: 875
type: RSZ, layer: 3, pos: 378
type: RSZ, layer: 3, pos: 289
type: RSZ, layer: 3, pos: 369
type: RSZ, layer: 3, pos: 893
type: RSZ, layer: 3, pos: 351
type: RSZ, layer: 3, pos: 991
type: RSZ, layer: 3, pos: 999
type: RSZ, layer: 3, pos: 988
type: RSZ, layer: 3, pos: 305
type: RSZ, layer: 3, pos: 871
type: RSZ, layer: 3, pos: 881
type: RSZ, layer: 3, pos: 996
type: RSZ, layer: 3, pos: 876
type: RSZ, layer: 3, pos: 383
type: RSZ, layer: 3, pos: 299
type: RSZ, layer: 3, pos: 361
type: RSZ, layer: 3, pos: 877
type: RSZ, layer: 3, pos: 1015
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 381
type: RSZ, layer: 3, pos: 291
type: RSZ, layer: 3, pos: 306
type: RSZ, layer: 3, pos: 993
type: RSZ, layer: 3, pos: 382
type: RSZ, layer: 3, pos: 851
type: RSZ, layer: 3, pos: 334
type: RSZ, layer: 3, pos: 843
type: RSZ, layer: 3, pos: 865
type: RSZ, layer: 3, pos: 314
type: RSZ, layer: 3, pos: 282
type: RSZ, layer: 3, pos: 858
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 859
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 889
type: RSZ, layer: 3, pos: 863
type: RSZ, layer: 3, pos: 895
type: RSZ, layer: 3, pos: 1009
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 274
type: RSZ, layer: 3, pos: 346
type: RSZ, layer: 3, pos: 283
type: RSZ, layer: 3, pos: 849
type: RSZ, layer: 3, pos: 1004
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 271
type: RSZ, layer: 3, pos: 379
type: RSZ, layer: 3, pos: 1023
type: RSZ, layer: 3, pos: 972
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 335
type: RSZ, layer: 3, pos: 850
type: RSZ, layer: 3, pos: 986
type: RSZ, layer: 3, pos: 279
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 700
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 980
type: RSZ, layer: 3, pos: 1005
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 884
type: RSZ, layer: 3, pos: 340
type: RSZ, layer: 3, pos: 1003
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 343
type: RSZ, layer: 3, pos: 846
type: RSZ, layer: 3, pos: 684
type: RSZ, layer: 3, pos: 689
type: RSZ, layer: 3, pos: 885
type: RSZ, layer: 3, pos: 315
type: RSZ, layer: 3, pos: 1020
type: RSZ, layer: 3, pos: 978
type: RSZ, layer: 3, pos: 995
type: RSZ, layer: 3, pos: 300
type: RSZ, layer: 3, pos: 331
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 319
type: RSZ, layer: 3, pos: 882
type: RSZ, layer: 3, pos: 364
type: RSZ, layer: 3, pos: 370
type: RSZ, layer: 3, pos: 84
type: RSZ, layer: 3, pos: 1021
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 673
type: RSZ, layer: 3, pos: 235
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 338
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 63
type: RSZ, layer: 3, pos: 699
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 58
type: RSZ, layer: 3, pos: 85
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 297
type: RSZ, layer: 3, pos: 239
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 339
type: RSZ, layer: 3, pos: 281
type: RSZ, layer: 3, pos: 69
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1019
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 344
type: RSZ, layer: 3, pos: 68
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 270
type: RSZ, layer: 3, pos: 883
type: RSZ, layer: 3, pos: 358
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 372
type: RSZ, layer: 3, pos: 666
type: RSZ, layer: 3, pos: 974
type: RSZ, layer: 3, pos: 667
type: RSZ, layer: 3, pos: 644
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 656
type: RSZ, layer: 3, pos: 870
type: RSZ, layer: 3, pos: 690
type: RSZ, layer: 3, pos: 273
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 336
type: RSZ, layer: 3, pos: 647
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 860
type: RSZ, layer: 3, pos: 841
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 318
type: RSZ, layer: 3, pos: 977
type: RSZ, layer: 3, pos: 61
type: RSZ, layer: 3, pos: 879
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 345
type: RSZ, layer: 3, pos: 663
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 110
type: RSZ, layer: 3, pos: 1017
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 102
type: RSZ, layer: 3, pos: 967
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 688
type: RSZ, layer: 3, pos: 50
type: RSZ, layer: 3, pos: 214
type: RSZ, layer: 3, pos: 867
type: RSZ, layer: 3, pos: 1014
type: RSZ, layer: 3, pos: 652
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 354
type: RSZ, layer: 3, pos: 114
type: RSZ, layer: 3, pos: 1012
type: RSZ, layer: 3, pos: 1018
type: RSZ, layer: 3, pos: 329
type: RSZ, layer: 3, pos: 844
type: RSZ, layer: 3, pos: 703
type: RSZ, layer: 3, pos: 326
type: RSZ, layer: 3, pos: 201
type: RSZ, layer: 3, pos: 376
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 1010
type: RSZ, layer: 3, pos: 259
type: RSZ, layer: 3, pos: 371
type: RSZ, layer: 3, pos: 874
type: RSZ, layer: 3, pos: 57
type: RSZ, layer: 3, pos: 272
type: RSZ, layer: 3, pos: 347
type: RSZ, layer: 3, pos: 692
type: RSZ, layer: 3, pos: 646
type: RSZ, layer: 3, pos: 275
type: RSZ, layer: 3, pos: 56
type: RSZ, layer: 3, pos: 203
type: RSZ, layer: 3, pos: 657
type: RSZ, layer: 3, pos: 420
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 276
type: RSZ, layer: 3, pos: 113
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 265
type: RSZ, layer: 3, pos: 419
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 210
type: RSZ, layer: 3, pos: 324
type: RSZ, layer: 3, pos: 123
type: RSZ, layer: 3, pos: 285
type: RSZ, layer: 3, pos: 965
type: RSZ, layer: 3, pos: 303
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 643
type: RSZ, layer: 3, pos: 847
type: RSZ, layer: 3, pos: 94
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 985
type: RSZ, layer: 3, pos: 658
type: RSZ, layer: 3, pos: 362
type: RSZ, layer: 3, pos: 665
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 674
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 360
type: RSZ, layer: 3, pos: 702
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 54
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 982
type: RSZ, layer: 3, pos: 866
type: RSZ, layer: 3, pos: 852
type: RSZ, layer: 3, pos: 1013
type: RSZ, layer: 3, pos: 51
type: RSZ, layer: 3, pos: 989
type: RSZ, layer: 3, pos: 833
type: RSZ, layer: 3, pos: 664
type: RSZ, layer: 3, pos: 82
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 628
type: RSZ, layer: 3, pos: 694
type: RSZ, layer: 3, pos: 53
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 349
type: RSZ, layer: 3, pos: 258
type: RSZ, layer: 3, pos: 428
type: RSZ, layer: 3, pos: 333
type: RSZ, layer: 3, pos: 105
type: RSZ, layer: 3, pos: 971
type: RSZ, layer: 3, pos: 365
type: RSZ, layer: 3, pos: 202
type: RSZ, layer: 3, pos: 266
type: RSZ, layer: 3, pos: 651
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 595
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 681
type: RSZ, layer: 3, pos: 683
type: RSZ, layer: 3, pos: 645
type: RSZ, layer: 3, pos: 421
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 296
type: RSZ, layer: 3, pos: 845
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 280
type: RSZ, layer: 3, pos: 1006
type: RSZ, layer: 3, pos: 1007
type: RSZ, layer: 3, pos: 648
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 970
type: RSZ, layer: 3, pos: 260
type: RSZ, layer: 3, pos: 857
type: RSZ, layer: 3, pos: 341
type: RSZ, layer: 3, pos: 320
type: RSZ, layer: 3, pos: 55
type: RSZ, layer: 3, pos: 973
type: RSZ, layer: 3, pos: 301
type: RSZ, layer: 3, pos: 328
type: RSZ, layer: 3, pos: 623
type: RSZ, layer: 3, pos: 78
type: RSZ, layer: 3, pos: 650
type: RSZ, layer: 3, pos: 1011
type: RSZ, layer: 3, pos: 321
type: RSZ, layer: 3, pos: 598
type: RSZ, layer: 3, pos: 127
type: RSZ, layer: 3, pos: 207
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 597
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 649
type: RSZ, layer: 3, pos: 304
type: RSZ, layer: 3, pos: 596
type: RSZ, layer: 3, pos: 682
type: RSZ, layer: 3, pos: 701
type: RSZ, layer: 3, pos: 86
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 247
type: RSZ, layer: 3, pos: 109
type: RSZ, layer: 3, pos: 853
type: RSZ, layer: 3, pos: 1002
type: RSZ, layer: 3, pos: 261
type: RSZ, layer: 3, pos: 672
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 862
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 855
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 687
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 593
type: RSZ, layer: 3, pos: 243
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 205
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 1008
type: RSZ, layer: 3, pos: 630
type: RSZ, layer: 3, pos: 126
type: RSZ, layer: 3, pos: 602
type: RSZ, layer: 3, pos: 215
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 589
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 60
type: RSZ, layer: 3, pos: 367
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 835
type: RSZ, layer: 3, pos: 642
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 586
type: RSZ, layer: 3, pos: 609
type: RSZ, layer: 3, pos: 352
type: RSZ, layer: 3, pos: 263
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 696
type: RSZ, layer: 3, pos: 590
type: RSZ, layer: 3, pos: 1022
type: RSZ, layer: 3, pos: 587
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 288
type: RSZ, layer: 3, pos: 842
type: RSZ, layer: 3, pos: 327
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 836
type: RSZ, layer: 3, pos: 990
type: RSZ, layer: 3, pos: 269
type: RSZ, layer: 3, pos: 52
type: RSZ, layer: 3, pos: 981
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 264
type: RSZ, layer: 3, pos: 631
type: RSZ, layer: 3, pos: 636
type: RSZ, layer: 3, pos: 256
type: RSZ, layer: 3, pos: 966
type: RSZ, layer: 3, pos: 246
type: RSZ, layer: 3, pos: 873
type: RSZ, layer: 3, pos: 968
type: RSZ, layer: 3, pos: 594
type: RSZ, layer: 3, pos: 413
type: RSZ, layer: 3, pos: 622
type: RSZ, layer: 3, pos: 119
type: RSZ, layer: 3, pos: 1016
type: RSZ, layer: 3, pos: 653
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 639
type: RSZ, layer: 3, pos: 585
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 987
type: RSZ, layer: 3, pos: 606
type: RSZ, layer: 3, pos: 979
type: RSZ, layer: 3, pos: 405
type: RSZ, layer: 3, pos: 998
type: RSZ, layer: 3, pos: 641
type: RSZ, layer: 3, pos: 257
type: RSZ, layer: 3, pos: 267
type: RSZ, layer: 3, pos: 217
type: RSZ, layer: 3, pos: 680
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 618
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 617
type: RSZ, layer: 3, pos: 1001
type: RSZ, layer: 3, pos: 834
type: RSZ, layer: 3, pos: 97
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 685
type: RSZ, layer: 3, pos: 125
type: RSZ, layer: 3, pos: 983
type: RSZ, layer: 3, pos: 591
type: RSZ, layer: 3, pos: 368
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 87
type: RSZ, layer: 3, pos: 607
type: RSZ, layer: 3, pos: 88
type: RSZ, layer: 3, pos: 322
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 960
type: RSZ, layer: 3, pos: 691
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 861
type: RSZ, layer: 3, pos: 89
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 74
type: RSZ, layer: 3, pos: 629
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 262
type: RSZ, layer: 3, pos: 112
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 659
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 209
type: RSZ, layer: 3, pos: 601
type: RSZ, layer: 3, pos: 588
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 580
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 868

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.0699769, upper bound: 14.5230048
time: 66.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.0765832, upper bound: 14.1007543
time: 56.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.5578117, -0.2815094, -37.5578117, -0.2815094, -37.2763023, 37.2763023
1: -17.6220856, 10.4812832, -17.6220856, 10.4812832, -28.1033688, 28.1033688
2: -14.4312439, 10.0883579, -14.4312439, 10.0883579, -24.5196018, 24.5196018
3: -14.9140787, 14.0766926, -14.9140787, 14.0766926, -28.9907722, 28.9907722
4: -15.1334782, 14.7552061, -15.1334782, 14.7552061, -29.8886833, 29.8886833
5: -14.2266178, 15.1941137, -14.2266178, 15.1941137, -29.4207306, 29.4207306
6: -20.8351784, 10.3227959, -20.8351784, 10.3227959, -31.1579742, 31.1579742
7: -17.3737793, 16.5286179, -17.3737793, 16.5286179, -33.4986191, 33.4999847
8: -16.3228550, 19.1706161, -16.3228550, 19.1706161, -35.4399414, 35.4383812
9: -15.1635761, 13.7782021, -15.1635761, 13.7782021, -28.7380333, 28.7390366
10: -23.5488243, 17.2546597, -23.5488243, 17.2546597, -40.8034821, 40.8034821
11: -26.2301464, 10.3602743, -26.2301464, 10.3602743, -36.5904198, 36.5904198
12: -24.2345219, 12.1104794, -24.2345219, 12.1104794, -36.3450012, 36.3450012
13: -22.1990852, 18.4545364, -22.1990852, 18.4545364, -40.6536217, 40.6536217
14: -47.8707161, -0.4218502, -47.8707161, -0.4218502, -47.2672577, 47.2434845
15: -19.6529541, 10.3302298, -19.6529541, 10.3302298, -29.9831848, 29.9831848
16: -24.9793587, 13.2617989, -24.9793587, 13.2617989, -37.7730865, 37.7719765
17: -43.9613876, 12.4236822, -43.9613876, 12.4236822, -55.0746613, 55.0541534
18: -20.4577484, 12.4874125, -20.4577484, 12.4874125, -32.9451599, 32.9451599
19: -17.9183464, 4.2855158, -17.9183464, 4.2855158, -22.2038612, 22.2038612
20: -15.2772102, 8.4781389, -15.2772102, 8.4781389, -23.7553482, 23.7553482
21: -25.8919182, 3.7857480, -25.8919182, 3.7857480, -29.6776657, 29.6776657
22: -32.9530487, -0.8713036, -32.9530487, -0.8713036, -30.6804123, 30.6864052
23: -17.9241028, 8.9438848, -17.9241028, 8.9438848, -26.8679886, 26.8679886
24: -25.2896576, 7.3427725, -25.2896576, 7.3427725, -31.1147385, 31.1136761
25: -18.3243713, 10.8286724, -18.3243713, 10.8286724, -29.1530437, 29.1530437
26: -23.7112923, 14.9050426, -23.7112923, 14.9050426, -38.6163330, 38.6163330
27: -26.2949791, 6.6883535, -26.2949791, 6.6883535, -31.9614639, 31.9720078
28: -17.3191872, 10.6421642, -17.3191872, 10.6421642, -27.7720261, 27.7722073
29: -40.1622734, -5.2830868, -40.1622734, -5.2830868, -33.8938408, 33.8988838
30: -20.8790817, 12.3533125, -20.8790817, 12.3533125, -33.2323952, 33.2323952
31: -23.7122688, 7.0026283, -23.7122688, 7.0026283, -30.7148972, 30.7148972
32: -27.6387272, 4.3835154, -27.6387272, 4.3835154, -31.0787277, 31.0826073
33: -30.5414295, 14.6008110, -30.5414295, 14.6008110, -44.2422104, 44.2393188
34: -25.9748363, 9.9279261, -25.9748363, 9.9279261, -35.9027634, 35.9027634
35: -27.7439537, 10.9692993, -27.7439537, 10.9692993, -38.3053703, 38.3007431
36: -27.1807117, 10.9039993, -27.1807117, 10.9039993, -37.6292725, 37.6310081
37: -37.2227592, 9.6494160, -37.2227592, 9.6494160, -45.5451050, 45.5455399
38: -29.7080078, 13.9970264, -29.7080078, 13.9970264, -43.7050323, 43.7050323
39: -38.4577255, 11.6211214, -38.4577255, 11.6211214, -49.4298172, 49.4302635
40: -30.4051437, 9.7967424, -30.4051437, 9.7967424, -38.5363770, 38.5354805
41: -22.3953133, 9.5435400, -22.3953133, 9.5435400, -31.9369202, 31.9388542
42: -16.3960018, 7.5732532, -16.3960018, 7.5732532, -23.6206646, 23.6249638

Time for backsubstitution: 0.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 868
type: RSZ, layer: 3, pos: 292
type: RSZ, layer: 3, pos: 236
type: RSZ, layer: 3, pos: 228
type: RSZ, layer: 3, pos: 355
type: RSZ, layer: 3, pos: 357
type: RSZ, layer: 3, pos: 284
type: RSZ, layer: 3, pos: 997
type: RSZ, layer: 3, pos: 724
type: RSZ, layer: 3, pos: 380
type: RSZ, layer: 3, pos: 363
type: RSZ, layer: 3, pos: 892
type: RSZ, layer: 3, pos: 375
type: RSZ, layer: 3, pos: 353
type: RSZ, layer: 3, pos: 377
type: RSZ, layer: 3, pos: 293
type: RSZ, layer: 3, pos: 887
type: RSZ, layer: 3, pos: 348
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 869
type: RSZ, layer: 3, pos: 316
type: RSZ, layer: 3, pos: 875
type: RSZ, layer: 3, pos: 378
type: RSZ, layer: 3, pos: 289
type: RSZ, layer: 3, pos: 369
type: RSZ, layer: 3, pos: 893
type: RSZ, layer: 3, pos: 351
type: RSZ, layer: 3, pos: 991
type: RSZ, layer: 3, pos: 999
type: RSZ, layer: 3, pos: 988
type: RSZ, layer: 3, pos: 305
type: RSZ, layer: 3, pos: 871
type: RSZ, layer: 3, pos: 881
type: RSZ, layer: 3, pos: 996
type: RSZ, layer: 3, pos: 876
type: RSZ, layer: 3, pos: 383
type: RSZ, layer: 3, pos: 299
type: RSZ, layer: 3, pos: 361
type: RSZ, layer: 3, pos: 877
type: RSZ, layer: 3, pos: 1015
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 381
type: RSZ, layer: 3, pos: 291
type: RSZ, layer: 3, pos: 306
type: RSZ, layer: 3, pos: 993
type: RSZ, layer: 3, pos: 382
type: RSZ, layer: 3, pos: 851
type: RSZ, layer: 3, pos: 334
type: RSZ, layer: 3, pos: 843
type: RSZ, layer: 3, pos: 865
type: RSZ, layer: 3, pos: 314
type: RSZ, layer: 3, pos: 282
type: RSZ, layer: 3, pos: 858
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 859
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 889
type: RSZ, layer: 3, pos: 863
type: RSZ, layer: 3, pos: 895
type: RSZ, layer: 3, pos: 1009
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 274
type: RSZ, layer: 3, pos: 346
type: RSZ, layer: 3, pos: 283
type: RSZ, layer: 3, pos: 849
type: RSZ, layer: 3, pos: 1004
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 271
type: RSZ, layer: 3, pos: 379
type: RSZ, layer: 3, pos: 1023
type: RSZ, layer: 3, pos: 972
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 335
type: RSZ, layer: 3, pos: 850
type: RSZ, layer: 3, pos: 986
type: RSZ, layer: 3, pos: 279
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 700
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 980
type: RSZ, layer: 3, pos: 1005
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 884
type: RSZ, layer: 3, pos: 340
type: RSZ, layer: 3, pos: 1003
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 343
type: RSZ, layer: 3, pos: 846
type: RSZ, layer: 3, pos: 684
type: RSZ, layer: 3, pos: 689
type: RSZ, layer: 3, pos: 885
type: RSZ, layer: 3, pos: 315
type: RSZ, layer: 3, pos: 1020
type: RSZ, layer: 3, pos: 978
type: RSZ, layer: 3, pos: 995
type: RSZ, layer: 3, pos: 300
type: RSZ, layer: 3, pos: 331
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 319
type: RSZ, layer: 3, pos: 882
type: RSZ, layer: 3, pos: 364
type: RSZ, layer: 3, pos: 370
type: RSZ, layer: 3, pos: 84
type: RSZ, layer: 3, pos: 1021
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 673
type: RSZ, layer: 3, pos: 235
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 338
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 63
type: RSZ, layer: 3, pos: 699
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 58
type: RSZ, layer: 3, pos: 85
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 297
type: RSZ, layer: 3, pos: 239
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 339
type: RSZ, layer: 3, pos: 281
type: RSZ, layer: 3, pos: 69
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1019
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 344
type: RSZ, layer: 3, pos: 68
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 270
type: RSZ, layer: 3, pos: 883
type: RSZ, layer: 3, pos: 358
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 372
type: RSZ, layer: 3, pos: 666
type: RSZ, layer: 3, pos: 974
type: RSZ, layer: 3, pos: 667
type: RSZ, layer: 3, pos: 644
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 656
type: RSZ, layer: 3, pos: 870
type: RSZ, layer: 3, pos: 690
type: RSZ, layer: 3, pos: 273
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 336
type: RSZ, layer: 3, pos: 647
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 860
type: RSZ, layer: 3, pos: 841
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 318
type: RSZ, layer: 3, pos: 977
type: RSZ, layer: 3, pos: 61
type: RSZ, layer: 3, pos: 879
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 345
type: RSZ, layer: 3, pos: 663
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 110
type: RSZ, layer: 3, pos: 1017
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 102
type: RSZ, layer: 3, pos: 967
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 688
type: RSZ, layer: 3, pos: 50
type: RSZ, layer: 3, pos: 214
type: RSZ, layer: 3, pos: 867
type: RSZ, layer: 3, pos: 1014
type: RSZ, layer: 3, pos: 652
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 354
type: RSZ, layer: 3, pos: 114
type: RSZ, layer: 3, pos: 1012
type: RSZ, layer: 3, pos: 1018
type: RSZ, layer: 3, pos: 329
type: RSZ, layer: 3, pos: 844
type: RSZ, layer: 3, pos: 703
type: RSZ, layer: 3, pos: 326
type: RSZ, layer: 3, pos: 201
type: RSZ, layer: 3, pos: 376
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 1010
type: RSZ, layer: 3, pos: 259
type: RSZ, layer: 3, pos: 371
type: RSZ, layer: 3, pos: 874
type: RSZ, layer: 3, pos: 57
type: RSZ, layer: 3, pos: 272
type: RSZ, layer: 3, pos: 347
type: RSZ, layer: 3, pos: 692
type: RSZ, layer: 3, pos: 646
type: RSZ, layer: 3, pos: 275
type: RSZ, layer: 3, pos: 56
type: RSZ, layer: 3, pos: 203
type: RSZ, layer: 3, pos: 657
type: RSZ, layer: 3, pos: 420
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 276
type: RSZ, layer: 3, pos: 113
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 265
type: RSZ, layer: 3, pos: 419
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 210
type: RSZ, layer: 3, pos: 324
type: RSZ, layer: 3, pos: 123
type: RSZ, layer: 3, pos: 285
type: RSZ, layer: 3, pos: 965
type: RSZ, layer: 3, pos: 303
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 643
type: RSZ, layer: 3, pos: 847
type: RSZ, layer: 3, pos: 94
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 985
type: RSZ, layer: 3, pos: 658
type: RSZ, layer: 3, pos: 362
type: RSZ, layer: 3, pos: 665
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 674
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 360
type: RSZ, layer: 3, pos: 702
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 54
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 982
type: RSZ, layer: 3, pos: 866
type: RSZ, layer: 3, pos: 852
type: RSZ, layer: 3, pos: 1013
type: RSZ, layer: 3, pos: 51
type: RSZ, layer: 3, pos: 989
type: RSZ, layer: 3, pos: 833
type: RSZ, layer: 3, pos: 664
type: RSZ, layer: 3, pos: 82
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 628
type: RSZ, layer: 3, pos: 694
type: RSZ, layer: 3, pos: 53
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 349
type: RSZ, layer: 3, pos: 258
type: RSZ, layer: 3, pos: 428
type: RSZ, layer: 3, pos: 333
type: RSZ, layer: 3, pos: 105
type: RSZ, layer: 3, pos: 971
type: RSZ, layer: 3, pos: 365
type: RSZ, layer: 3, pos: 202
type: RSZ, layer: 3, pos: 266
type: RSZ, layer: 3, pos: 651
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 595
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 681
type: RSZ, layer: 3, pos: 683
type: RSZ, layer: 3, pos: 645
type: RSZ, layer: 3, pos: 421
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 296
type: RSZ, layer: 3, pos: 845
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 280
type: RSZ, layer: 3, pos: 1006
type: RSZ, layer: 3, pos: 1007
type: RSZ, layer: 3, pos: 648
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 970
type: RSZ, layer: 3, pos: 260
type: RSZ, layer: 3, pos: 857
type: RSZ, layer: 3, pos: 341
type: RSZ, layer: 3, pos: 320
type: RSZ, layer: 3, pos: 55
type: RSZ, layer: 3, pos: 973
type: RSZ, layer: 3, pos: 301
type: RSZ, layer: 3, pos: 328
type: RSZ, layer: 3, pos: 623
type: RSZ, layer: 3, pos: 78
type: RSZ, layer: 3, pos: 650
type: RSZ, layer: 3, pos: 1011
type: RSZ, layer: 3, pos: 321
type: RSZ, layer: 3, pos: 598
type: RSZ, layer: 3, pos: 127
type: RSZ, layer: 3, pos: 207
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 597
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 649
type: RSZ, layer: 3, pos: 304
type: RSZ, layer: 3, pos: 596
type: RSZ, layer: 3, pos: 682
type: RSZ, layer: 3, pos: 701
type: RSZ, layer: 3, pos: 86
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 247
type: RSZ, layer: 3, pos: 109
type: RSZ, layer: 3, pos: 853
type: RSZ, layer: 3, pos: 1002
type: RSZ, layer: 3, pos: 261
type: RSZ, layer: 3, pos: 672
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 862
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 855
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 687
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 593
type: RSZ, layer: 3, pos: 243
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 205
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 1008
type: RSZ, layer: 3, pos: 630
type: RSZ, layer: 3, pos: 126
type: RSZ, layer: 3, pos: 602
type: RSZ, layer: 3, pos: 215
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 589
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 60
type: RSZ, layer: 3, pos: 367
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 835
type: RSZ, layer: 3, pos: 642
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 586
type: RSZ, layer: 3, pos: 609
type: RSZ, layer: 3, pos: 352
type: RSZ, layer: 3, pos: 263
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 696
type: RSZ, layer: 3, pos: 590
type: RSZ, layer: 3, pos: 1022
type: RSZ, layer: 3, pos: 587
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 288
type: RSZ, layer: 3, pos: 842
type: RSZ, layer: 3, pos: 327
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 836
type: RSZ, layer: 3, pos: 990
type: RSZ, layer: 3, pos: 269
type: RSZ, layer: 3, pos: 52
type: RSZ, layer: 3, pos: 981
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 264
type: RSZ, layer: 3, pos: 631
type: RSZ, layer: 3, pos: 636
type: RSZ, layer: 3, pos: 256
type: RSZ, layer: 3, pos: 966
type: RSZ, layer: 3, pos: 246
type: RSZ, layer: 3, pos: 873
type: RSZ, layer: 3, pos: 968
type: RSZ, layer: 3, pos: 594
type: RSZ, layer: 3, pos: 413
type: RSZ, layer: 3, pos: 622
type: RSZ, layer: 3, pos: 119
type: RSZ, layer: 3, pos: 1016
type: RSZ, layer: 3, pos: 653
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 639
type: RSZ, layer: 3, pos: 585
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 987
type: RSZ, layer: 3, pos: 606
type: RSZ, layer: 3, pos: 979
type: RSZ, layer: 3, pos: 405
type: RSZ, layer: 3, pos: 998
type: RSZ, layer: 3, pos: 641
type: RSZ, layer: 3, pos: 257
type: RSZ, layer: 3, pos: 267
type: RSZ, layer: 3, pos: 217
type: RSZ, layer: 3, pos: 680
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 618
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 617
type: RSZ, layer: 3, pos: 1001
type: RSZ, layer: 3, pos: 834
type: RSZ, layer: 3, pos: 97
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 685
type: RSZ, layer: 3, pos: 125
type: RSZ, layer: 3, pos: 983
type: RSZ, layer: 3, pos: 591
type: RSZ, layer: 3, pos: 368
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 87
type: RSZ, layer: 3, pos: 607
type: RSZ, layer: 3, pos: 88
type: RSZ, layer: 3, pos: 322
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 960
type: RSZ, layer: 3, pos: 691
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 861
type: RSZ, layer: 3, pos: 89
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 74
type: RSZ, layer: 3, pos: 629
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 262
type: RSZ, layer: 3, pos: 112
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 659
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 209
type: RSZ, layer: 3, pos: 601
type: RSZ, layer: 3, pos: 588
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 580
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 868

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1549993, upper bound: 14.3802180
time: 57.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1669745, upper bound: 14.0018900
time: 62.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.5578117, -0.2815094, -37.5578117, -0.2815094, -37.2763023, 37.2763023
1: -17.6220856, 10.4812832, -17.6220856, 10.4812832, -28.1033688, 28.1033688
2: -14.4312439, 10.0883579, -14.4312439, 10.0883579, -24.5196018, 24.5196018
3: -14.9140787, 14.0766926, -14.9140787, 14.0766926, -28.9907722, 28.9907722
4: -15.1334782, 14.7552061, -15.1334782, 14.7552061, -29.8886833, 29.8886833
5: -14.2266178, 15.1941137, -14.2266178, 15.1941137, -29.4207306, 29.4207306
6: -20.8351784, 10.3227959, -20.8351784, 10.3227959, -31.1579742, 31.1579742
7: -17.3737793, 16.5286179, -17.3737793, 16.5286179, -33.4999771, 33.4986267
8: -16.3228550, 19.1706161, -16.3228550, 19.1706161, -35.4383850, 35.4399414
9: -15.1635761, 13.7782021, -15.1635761, 13.7782021, -28.7390404, 28.7380295
10: -23.5488243, 17.2546597, -23.5488243, 17.2546597, -40.8034821, 40.8034821
11: -26.2301464, 10.3602743, -26.2301464, 10.3602743, -36.5904198, 36.5904198
12: -24.2345219, 12.1104794, -24.2345219, 12.1104794, -36.3450012, 36.3450012
13: -22.1990852, 18.4545364, -22.1990852, 18.4545364, -40.6536217, 40.6536217
14: -47.8707161, -0.4218502, -47.8707161, -0.4218502, -47.2434845, 47.2672615
15: -19.6529541, 10.3302298, -19.6529541, 10.3302298, -29.9831848, 29.9831848
16: -24.9793587, 13.2617989, -24.9793587, 13.2617989, -37.7719727, 37.7730865
17: -43.9613876, 12.4236822, -43.9613876, 12.4236822, -55.0541687, 55.0746536
18: -20.4577484, 12.4874125, -20.4577484, 12.4874125, -32.9451599, 32.9451599
19: -17.9183464, 4.2855158, -17.9183464, 4.2855158, -22.2038612, 22.2038612
20: -15.2772102, 8.4781389, -15.2772102, 8.4781389, -23.7553482, 23.7553482
21: -25.8919182, 3.7857480, -25.8919182, 3.7857480, -29.6776657, 29.6776657
22: -32.9530487, -0.8713036, -32.9530487, -0.8713036, -30.6864090, 30.6804047
23: -17.9241028, 8.9438848, -17.9241028, 8.9438848, -26.8679886, 26.8679886
24: -25.2896576, 7.3427725, -25.2896576, 7.3427725, -31.1136780, 31.1147423
25: -18.3243713, 10.8286724, -18.3243713, 10.8286724, -29.1530437, 29.1530437
26: -23.7112923, 14.9050426, -23.7112923, 14.9050426, -38.6163330, 38.6163330
27: -26.2949791, 6.6883535, -26.2949791, 6.6883535, -31.9720078, 31.9614601
28: -17.3191872, 10.6421642, -17.3191872, 10.6421642, -27.7722092, 27.7720261
29: -40.1622734, -5.2830868, -40.1622734, -5.2830868, -33.8988838, 33.8938370
30: -20.8790817, 12.3533125, -20.8790817, 12.3533125, -33.2323952, 33.2323952
31: -23.7122688, 7.0026283, -23.7122688, 7.0026283, -30.7148972, 30.7148972
32: -27.6387272, 4.3835154, -27.6387272, 4.3835154, -31.0826035, 31.0787315
33: -30.5414295, 14.6008110, -30.5414295, 14.6008110, -44.2393188, 44.2422104
34: -25.9748363, 9.9279261, -25.9748363, 9.9279261, -35.9027634, 35.9027634
35: -27.7439537, 10.9692993, -27.7439537, 10.9692993, -38.3007469, 38.3053665
36: -27.1807117, 10.9039993, -27.1807117, 10.9039993, -37.6310120, 37.6292725
37: -37.2227592, 9.6494160, -37.2227592, 9.6494160, -45.5455399, 45.5451050
38: -29.7080078, 13.9970264, -29.7080078, 13.9970264, -43.7050323, 43.7050323
39: -38.4577255, 11.6211214, -38.4577255, 11.6211214, -49.4302597, 49.4298172
40: -30.4051437, 9.7967424, -30.4051437, 9.7967424, -38.5354767, 38.5363770
41: -22.3953133, 9.5435400, -22.3953133, 9.5435400, -31.9388542, 31.9369240
42: -16.3960018, 7.5732532, -16.3960018, 7.5732532, -23.6249676, 23.6206646

Time for backsubstitution: 0.90 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 868
type: RSZ, layer: 3, pos: 292
type: RSZ, layer: 3, pos: 236
type: RSZ, layer: 3, pos: 228
type: RSZ, layer: 3, pos: 355
type: RSZ, layer: 3, pos: 357
type: RSZ, layer: 3, pos: 284
type: RSZ, layer: 3, pos: 997
type: RSZ, layer: 3, pos: 724
type: RSZ, layer: 3, pos: 380
type: RSZ, layer: 3, pos: 363
type: RSZ, layer: 3, pos: 892
type: RSZ, layer: 3, pos: 375
type: RSZ, layer: 3, pos: 353
type: RSZ, layer: 3, pos: 377
type: RSZ, layer: 3, pos: 293
type: RSZ, layer: 3, pos: 887
type: RSZ, layer: 3, pos: 348
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 869
type: RSZ, layer: 3, pos: 316
type: RSZ, layer: 3, pos: 875
type: RSZ, layer: 3, pos: 378
type: RSZ, layer: 3, pos: 289
type: RSZ, layer: 3, pos: 369
type: RSZ, layer: 3, pos: 893
type: RSZ, layer: 3, pos: 351
type: RSZ, layer: 3, pos: 991
type: RSZ, layer: 3, pos: 999
type: RSZ, layer: 3, pos: 988
type: RSZ, layer: 3, pos: 305
type: RSZ, layer: 3, pos: 871
type: RSZ, layer: 3, pos: 881
type: RSZ, layer: 3, pos: 996
type: RSZ, layer: 3, pos: 876
type: RSZ, layer: 3, pos: 383
type: RSZ, layer: 3, pos: 299
type: RSZ, layer: 3, pos: 361
type: RSZ, layer: 3, pos: 877
type: RSZ, layer: 3, pos: 1015
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 381
type: RSZ, layer: 3, pos: 291
type: RSZ, layer: 3, pos: 306
type: RSZ, layer: 3, pos: 993
type: RSZ, layer: 3, pos: 382
type: RSZ, layer: 3, pos: 851
type: RSZ, layer: 3, pos: 334
type: RSZ, layer: 3, pos: 843
type: RSZ, layer: 3, pos: 865
type: RSZ, layer: 3, pos: 314
type: RSZ, layer: 3, pos: 282
type: RSZ, layer: 3, pos: 858
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 859
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 889
type: RSZ, layer: 3, pos: 863
type: RSZ, layer: 3, pos: 895
type: RSZ, layer: 3, pos: 1009
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 274
type: RSZ, layer: 3, pos: 346
type: RSZ, layer: 3, pos: 283
type: RSZ, layer: 3, pos: 849
type: RSZ, layer: 3, pos: 1004
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 271
type: RSZ, layer: 3, pos: 379
type: RSZ, layer: 3, pos: 1023
type: RSZ, layer: 3, pos: 972
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 335
type: RSZ, layer: 3, pos: 850
type: RSZ, layer: 3, pos: 986
type: RSZ, layer: 3, pos: 279
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 700
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 980
type: RSZ, layer: 3, pos: 1005
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 884
type: RSZ, layer: 3, pos: 340
type: RSZ, layer: 3, pos: 1003
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 343
type: RSZ, layer: 3, pos: 846
type: RSZ, layer: 3, pos: 684
type: RSZ, layer: 3, pos: 689
type: RSZ, layer: 3, pos: 885
type: RSZ, layer: 3, pos: 315
type: RSZ, layer: 3, pos: 1020
type: RSZ, layer: 3, pos: 978
type: RSZ, layer: 3, pos: 995
type: RSZ, layer: 3, pos: 300
type: RSZ, layer: 3, pos: 331
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 319
type: RSZ, layer: 3, pos: 882
type: RSZ, layer: 3, pos: 364
type: RSZ, layer: 3, pos: 370
type: RSZ, layer: 3, pos: 84
type: RSZ, layer: 3, pos: 1021
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 673
type: RSZ, layer: 3, pos: 235
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 338
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 63
type: RSZ, layer: 3, pos: 699
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 58
type: RSZ, layer: 3, pos: 85
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 297
type: RSZ, layer: 3, pos: 239
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 339
type: RSZ, layer: 3, pos: 281
type: RSZ, layer: 3, pos: 69
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1019
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 344
type: RSZ, layer: 3, pos: 68
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 270
type: RSZ, layer: 3, pos: 883
type: RSZ, layer: 3, pos: 358
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 372
type: RSZ, layer: 3, pos: 666
type: RSZ, layer: 3, pos: 974
type: RSZ, layer: 3, pos: 667
type: RSZ, layer: 3, pos: 644
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 656
type: RSZ, layer: 3, pos: 870
type: RSZ, layer: 3, pos: 690
type: RSZ, layer: 3, pos: 273
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 336
type: RSZ, layer: 3, pos: 647
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 860
type: RSZ, layer: 3, pos: 841
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 318
type: RSZ, layer: 3, pos: 977
type: RSZ, layer: 3, pos: 61
type: RSZ, layer: 3, pos: 879
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 345
type: RSZ, layer: 3, pos: 663
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 110
type: RSZ, layer: 3, pos: 1017
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 102
type: RSZ, layer: 3, pos: 967
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 688
type: RSZ, layer: 3, pos: 50
type: RSZ, layer: 3, pos: 214
type: RSZ, layer: 3, pos: 867
type: RSZ, layer: 3, pos: 1014
type: RSZ, layer: 3, pos: 652
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 354
type: RSZ, layer: 3, pos: 114
type: RSZ, layer: 3, pos: 1012
type: RSZ, layer: 3, pos: 1018
type: RSZ, layer: 3, pos: 329
type: RSZ, layer: 3, pos: 844
type: RSZ, layer: 3, pos: 703
type: RSZ, layer: 3, pos: 326
type: RSZ, layer: 3, pos: 201
type: RSZ, layer: 3, pos: 376
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 1010
type: RSZ, layer: 3, pos: 259
type: RSZ, layer: 3, pos: 371
type: RSZ, layer: 3, pos: 874
type: RSZ, layer: 3, pos: 57
type: RSZ, layer: 3, pos: 272
type: RSZ, layer: 3, pos: 347
type: RSZ, layer: 3, pos: 692
type: RSZ, layer: 3, pos: 646
type: RSZ, layer: 3, pos: 275
type: RSZ, layer: 3, pos: 56
type: RSZ, layer: 3, pos: 203
type: RSZ, layer: 3, pos: 657
type: RSZ, layer: 3, pos: 420
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 276
type: RSZ, layer: 3, pos: 113
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 265
type: RSZ, layer: 3, pos: 419
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 210
type: RSZ, layer: 3, pos: 324
type: RSZ, layer: 3, pos: 123
type: RSZ, layer: 3, pos: 285
type: RSZ, layer: 3, pos: 965
type: RSZ, layer: 3, pos: 303
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 643
type: RSZ, layer: 3, pos: 847
type: RSZ, layer: 3, pos: 94
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 985
type: RSZ, layer: 3, pos: 658
type: RSZ, layer: 3, pos: 362
type: RSZ, layer: 3, pos: 665
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 674
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 360
type: RSZ, layer: 3, pos: 702
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 54
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 982
type: RSZ, layer: 3, pos: 866
type: RSZ, layer: 3, pos: 852
type: RSZ, layer: 3, pos: 1013
type: RSZ, layer: 3, pos: 51
type: RSZ, layer: 3, pos: 989
type: RSZ, layer: 3, pos: 833
type: RSZ, layer: 3, pos: 664
type: RSZ, layer: 3, pos: 82
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 628
type: RSZ, layer: 3, pos: 694
type: RSZ, layer: 3, pos: 53
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 349
type: RSZ, layer: 3, pos: 258
type: RSZ, layer: 3, pos: 428
type: RSZ, layer: 3, pos: 333
type: RSZ, layer: 3, pos: 105
type: RSZ, layer: 3, pos: 971
type: RSZ, layer: 3, pos: 365
type: RSZ, layer: 3, pos: 202
type: RSZ, layer: 3, pos: 266
type: RSZ, layer: 3, pos: 651
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 595
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 681
type: RSZ, layer: 3, pos: 683
type: RSZ, layer: 3, pos: 645
type: RSZ, layer: 3, pos: 421
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 296
type: RSZ, layer: 3, pos: 845
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 280
type: RSZ, layer: 3, pos: 1006
type: RSZ, layer: 3, pos: 1007
type: RSZ, layer: 3, pos: 648
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 970
type: RSZ, layer: 3, pos: 260
type: RSZ, layer: 3, pos: 857
type: RSZ, layer: 3, pos: 341
type: RSZ, layer: 3, pos: 320
type: RSZ, layer: 3, pos: 55
type: RSZ, layer: 3, pos: 973
type: RSZ, layer: 3, pos: 301
type: RSZ, layer: 3, pos: 328
type: RSZ, layer: 3, pos: 623
type: RSZ, layer: 3, pos: 78
type: RSZ, layer: 3, pos: 650
type: RSZ, layer: 3, pos: 1011
type: RSZ, layer: 3, pos: 321
type: RSZ, layer: 3, pos: 598
type: RSZ, layer: 3, pos: 127
type: RSZ, layer: 3, pos: 207
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 597
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 649
type: RSZ, layer: 3, pos: 304
type: RSZ, layer: 3, pos: 596
type: RSZ, layer: 3, pos: 682
type: RSZ, layer: 3, pos: 701
type: RSZ, layer: 3, pos: 86
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 247
type: RSZ, layer: 3, pos: 109
type: RSZ, layer: 3, pos: 853
type: RSZ, layer: 3, pos: 1002
type: RSZ, layer: 3, pos: 261
type: RSZ, layer: 3, pos: 672
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 862
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 855
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 687
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 593
type: RSZ, layer: 3, pos: 243
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 205
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 1008
type: RSZ, layer: 3, pos: 630
type: RSZ, layer: 3, pos: 126
type: RSZ, layer: 3, pos: 602
type: RSZ, layer: 3, pos: 215
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 589
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 60
type: RSZ, layer: 3, pos: 367
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 835
type: RSZ, layer: 3, pos: 642
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 586
type: RSZ, layer: 3, pos: 609
type: RSZ, layer: 3, pos: 352
type: RSZ, layer: 3, pos: 263
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 696
type: RSZ, layer: 3, pos: 590
type: RSZ, layer: 3, pos: 1022
type: RSZ, layer: 3, pos: 587
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 288
type: RSZ, layer: 3, pos: 842
type: RSZ, layer: 3, pos: 327
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 836
type: RSZ, layer: 3, pos: 990
type: RSZ, layer: 3, pos: 269
type: RSZ, layer: 3, pos: 52
type: RSZ, layer: 3, pos: 981
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 264
type: RSZ, layer: 3, pos: 631
type: RSZ, layer: 3, pos: 636
type: RSZ, layer: 3, pos: 256
type: RSZ, layer: 3, pos: 966
type: RSZ, layer: 3, pos: 246
type: RSZ, layer: 3, pos: 873
type: RSZ, layer: 3, pos: 968
type: RSZ, layer: 3, pos: 594
type: RSZ, layer: 3, pos: 413
type: RSZ, layer: 3, pos: 622
type: RSZ, layer: 3, pos: 119
type: RSZ, layer: 3, pos: 1016
type: RSZ, layer: 3, pos: 653
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 639
type: RSZ, layer: 3, pos: 585
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 987
type: RSZ, layer: 3, pos: 606
type: RSZ, layer: 3, pos: 979
type: RSZ, layer: 3, pos: 405
type: RSZ, layer: 3, pos: 998
type: RSZ, layer: 3, pos: 641
type: RSZ, layer: 3, pos: 257
type: RSZ, layer: 3, pos: 267
type: RSZ, layer: 3, pos: 217
type: RSZ, layer: 3, pos: 680
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 618
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 617
type: RSZ, layer: 3, pos: 1001
type: RSZ, layer: 3, pos: 834
type: RSZ, layer: 3, pos: 97
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 685
type: RSZ, layer: 3, pos: 125
type: RSZ, layer: 3, pos: 983
type: RSZ, layer: 3, pos: 591
type: RSZ, layer: 3, pos: 368
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 87
type: RSZ, layer: 3, pos: 607
type: RSZ, layer: 3, pos: 88
type: RSZ, layer: 3, pos: 322
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 960
type: RSZ, layer: 3, pos: 691
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 861
type: RSZ, layer: 3, pos: 89
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 74
type: RSZ, layer: 3, pos: 629
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 262
type: RSZ, layer: 3, pos: 112
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 659
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 209
type: RSZ, layer: 3, pos: 601
type: RSZ, layer: 3, pos: 588
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 580
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 868

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.0018900, upper bound: 14.1669745
time: 46.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.3802180, upper bound: 14.1549993
time: 57.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.5578117, -0.2815094, -37.5578117, -0.2815094, -37.2763023, 37.2763023
1: -17.6220856, 10.4812832, -17.6220856, 10.4812832, -28.1033688, 28.1033688
2: -14.4312439, 10.0883579, -14.4312439, 10.0883579, -24.5196018, 24.5196018
3: -14.9140787, 14.0766926, -14.9140787, 14.0766926, -28.9907722, 28.9907722
4: -15.1334782, 14.7552061, -15.1334782, 14.7552061, -29.8886833, 29.8886833
5: -14.2266178, 15.1941137, -14.2266178, 15.1941137, -29.4207306, 29.4207306
6: -20.8351784, 10.3227959, -20.8351784, 10.3227959, -31.1579742, 31.1579742
7: -17.3737793, 16.5286179, -17.3737793, 16.5286179, -33.5006332, 33.4983025
8: -16.3228550, 19.1706161, -16.3228550, 19.1706161, -35.4403687, 35.4382133
9: -15.1635761, 13.7782021, -15.1635761, 13.7782021, -28.7402153, 28.7369270
10: -23.5488243, 17.2546597, -23.5488243, 17.2546597, -40.8034821, 40.8034821
11: -26.2301464, 10.3602743, -26.2301464, 10.3602743, -36.5904198, 36.5904198
12: -24.2345219, 12.1104794, -24.2345219, 12.1104794, -36.3450012, 36.3450012
13: -22.1990852, 18.4545364, -22.1990852, 18.4545364, -40.6536217, 40.6536217
14: -47.8707161, -0.4218502, -47.8707161, -0.4218502, -47.2433472, 47.2676544
15: -19.6529541, 10.3302298, -19.6529541, 10.3302298, -29.9831848, 29.9831848
16: -24.9793587, 13.2617989, -24.9793587, 13.2617989, -37.7770462, 37.7681465
17: -43.9613876, 12.4236822, -43.9613876, 12.4236822, -55.0533447, 55.0755463
18: -20.4577484, 12.4874125, -20.4577484, 12.4874125, -32.9451599, 32.9451599
19: -17.9183464, 4.2855158, -17.9183464, 4.2855158, -22.2038612, 22.2038612
20: -15.2772102, 8.4781389, -15.2772102, 8.4781389, -23.7553482, 23.7553482
21: -25.8919182, 3.7857480, -25.8919182, 3.7857480, -29.6776657, 29.6776657
22: -32.9530487, -0.8713036, -32.9530487, -0.8713036, -30.6773453, 30.6899643
23: -17.9241028, 8.9438848, -17.9241028, 8.9438848, -26.8679886, 26.8679886
24: -25.2896576, 7.3427725, -25.2896576, 7.3427725, -31.1083755, 31.1202850
25: -18.3243713, 10.8286724, -18.3243713, 10.8286724, -29.1530437, 29.1530437
26: -23.7112923, 14.9050426, -23.7112923, 14.9050426, -38.6163330, 38.6163330
27: -26.2949791, 6.6883535, -26.2949791, 6.6883535, -31.9714813, 31.9622192
28: -17.3191872, 10.6421642, -17.3191872, 10.6421642, -27.7719116, 27.7723236
29: -40.1622734, -5.2830868, -40.1622734, -5.2830868, -33.8892632, 33.9035797
30: -20.8790817, 12.3533125, -20.8790817, 12.3533125, -33.2323952, 33.2323952
31: -23.7122688, 7.0026283, -23.7122688, 7.0026283, -30.7148972, 30.7148972
32: -27.6387272, 4.3835154, -27.6387272, 4.3835154, -31.0828476, 31.0784969
33: -30.5414295, 14.6008110, -30.5414295, 14.6008110, -44.2429352, 44.2389832
34: -25.9748363, 9.9279261, -25.9748363, 9.9279261, -35.9027634, 35.9027634
35: -27.7439537, 10.9692993, -27.7439537, 10.9692993, -38.3038139, 38.3028526
36: -27.1807117, 10.9039993, -27.1807117, 10.9039993, -37.6312103, 37.6290894
37: -37.2227592, 9.6494160, -37.2227592, 9.6494160, -45.5470200, 45.5436401
38: -29.7080078, 13.9970264, -29.7080078, 13.9970264, -43.7050323, 43.7050323
39: -38.4577255, 11.6211214, -38.4577255, 11.6211214, -49.4291992, 49.4308739
40: -30.4051437, 9.7967424, -30.4051437, 9.7967424, -38.5360260, 38.5358620
41: -22.3953133, 9.5435400, -22.3953133, 9.5435400, -31.9388542, 31.9365158
42: -16.3960018, 7.5732532, -16.3960018, 7.5732532, -23.6266384, 23.6195107

Time for backsubstitution: 0.91 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 868
type: RSZ, layer: 3, pos: 292
type: RSZ, layer: 3, pos: 236
type: RSZ, layer: 3, pos: 228
type: RSZ, layer: 3, pos: 355
type: RSZ, layer: 3, pos: 357
type: RSZ, layer: 3, pos: 284
type: RSZ, layer: 3, pos: 997
type: RSZ, layer: 3, pos: 724
type: RSZ, layer: 3, pos: 380
type: RSZ, layer: 3, pos: 363
type: RSZ, layer: 3, pos: 892
type: RSZ, layer: 3, pos: 375
type: RSZ, layer: 3, pos: 353
type: RSZ, layer: 3, pos: 377
type: RSZ, layer: 3, pos: 293
type: RSZ, layer: 3, pos: 887
type: RSZ, layer: 3, pos: 348
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 869
type: RSZ, layer: 3, pos: 316
type: RSZ, layer: 3, pos: 875
type: RSZ, layer: 3, pos: 378
type: RSZ, layer: 3, pos: 289
type: RSZ, layer: 3, pos: 369
type: RSZ, layer: 3, pos: 893
type: RSZ, layer: 3, pos: 351
type: RSZ, layer: 3, pos: 991
type: RSZ, layer: 3, pos: 999
type: RSZ, layer: 3, pos: 988
type: RSZ, layer: 3, pos: 305
type: RSZ, layer: 3, pos: 871
type: RSZ, layer: 3, pos: 881
type: RSZ, layer: 3, pos: 996
type: RSZ, layer: 3, pos: 876
type: RSZ, layer: 3, pos: 383
type: RSZ, layer: 3, pos: 299
type: RSZ, layer: 3, pos: 361
type: RSZ, layer: 3, pos: 877
type: RSZ, layer: 3, pos: 1015
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 381
type: RSZ, layer: 3, pos: 291
type: RSZ, layer: 3, pos: 306
type: RSZ, layer: 3, pos: 993
type: RSZ, layer: 3, pos: 382
type: RSZ, layer: 3, pos: 851
type: RSZ, layer: 3, pos: 334
type: RSZ, layer: 3, pos: 843
type: RSZ, layer: 3, pos: 865
type: RSZ, layer: 3, pos: 314
type: RSZ, layer: 3, pos: 282
type: RSZ, layer: 3, pos: 858
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 859
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 889
type: RSZ, layer: 3, pos: 863
type: RSZ, layer: 3, pos: 895
type: RSZ, layer: 3, pos: 1009
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 274
type: RSZ, layer: 3, pos: 346
type: RSZ, layer: 3, pos: 283
type: RSZ, layer: 3, pos: 849
type: RSZ, layer: 3, pos: 1004
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 271
type: RSZ, layer: 3, pos: 379
type: RSZ, layer: 3, pos: 1023
type: RSZ, layer: 3, pos: 972
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 335
type: RSZ, layer: 3, pos: 850
type: RSZ, layer: 3, pos: 986
type: RSZ, layer: 3, pos: 279
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 700
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 980
type: RSZ, layer: 3, pos: 1005
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 884
type: RSZ, layer: 3, pos: 340
type: RSZ, layer: 3, pos: 1003
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 343
type: RSZ, layer: 3, pos: 846
type: RSZ, layer: 3, pos: 684
type: RSZ, layer: 3, pos: 689
type: RSZ, layer: 3, pos: 885
type: RSZ, layer: 3, pos: 315
type: RSZ, layer: 3, pos: 1020
type: RSZ, layer: 3, pos: 978
type: RSZ, layer: 3, pos: 995
type: RSZ, layer: 3, pos: 300
type: RSZ, layer: 3, pos: 331
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 319
type: RSZ, layer: 3, pos: 882
type: RSZ, layer: 3, pos: 364
type: RSZ, layer: 3, pos: 370
type: RSZ, layer: 3, pos: 84
type: RSZ, layer: 3, pos: 1021
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 673
type: RSZ, layer: 3, pos: 235
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 338
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 63
type: RSZ, layer: 3, pos: 699
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 58
type: RSZ, layer: 3, pos: 85
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 297
type: RSZ, layer: 3, pos: 239
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 339
type: RSZ, layer: 3, pos: 281
type: RSZ, layer: 3, pos: 69
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1019
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 344
type: RSZ, layer: 3, pos: 68
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 270
type: RSZ, layer: 3, pos: 883
type: RSZ, layer: 3, pos: 358
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 372
type: RSZ, layer: 3, pos: 666
type: RSZ, layer: 3, pos: 974
type: RSZ, layer: 3, pos: 667
type: RSZ, layer: 3, pos: 644
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 656
type: RSZ, layer: 3, pos: 870
type: RSZ, layer: 3, pos: 690
type: RSZ, layer: 3, pos: 273
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 336
type: RSZ, layer: 3, pos: 647
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 860
type: RSZ, layer: 3, pos: 841
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 318
type: RSZ, layer: 3, pos: 977
type: RSZ, layer: 3, pos: 61
type: RSZ, layer: 3, pos: 879
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 345
type: RSZ, layer: 3, pos: 663
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 110
type: RSZ, layer: 3, pos: 1017
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 102
type: RSZ, layer: 3, pos: 967
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 688
type: RSZ, layer: 3, pos: 50
type: RSZ, layer: 3, pos: 214
type: RSZ, layer: 3, pos: 867
type: RSZ, layer: 3, pos: 1014
type: RSZ, layer: 3, pos: 652
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 354
type: RSZ, layer: 3, pos: 114
type: RSZ, layer: 3, pos: 1012
type: RSZ, layer: 3, pos: 1018
type: RSZ, layer: 3, pos: 329
type: RSZ, layer: 3, pos: 844
type: RSZ, layer: 3, pos: 703
type: RSZ, layer: 3, pos: 326
type: RSZ, layer: 3, pos: 201
type: RSZ, layer: 3, pos: 376
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 1010
type: RSZ, layer: 3, pos: 259
type: RSZ, layer: 3, pos: 371
type: RSZ, layer: 3, pos: 874
type: RSZ, layer: 3, pos: 57
type: RSZ, layer: 3, pos: 272
type: RSZ, layer: 3, pos: 347
type: RSZ, layer: 3, pos: 692
type: RSZ, layer: 3, pos: 646
type: RSZ, layer: 3, pos: 275
type: RSZ, layer: 3, pos: 56
type: RSZ, layer: 3, pos: 203
type: RSZ, layer: 3, pos: 657
type: RSZ, layer: 3, pos: 420
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 276
type: RSZ, layer: 3, pos: 113
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 265
type: RSZ, layer: 3, pos: 419
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 210
type: RSZ, layer: 3, pos: 324
type: RSZ, layer: 3, pos: 123
type: RSZ, layer: 3, pos: 285
type: RSZ, layer: 3, pos: 965
type: RSZ, layer: 3, pos: 303
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 643
type: RSZ, layer: 3, pos: 847
type: RSZ, layer: 3, pos: 94
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 985
type: RSZ, layer: 3, pos: 658
type: RSZ, layer: 3, pos: 362
type: RSZ, layer: 3, pos: 665
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 674
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 360
type: RSZ, layer: 3, pos: 702
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 54
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 982
type: RSZ, layer: 3, pos: 866
type: RSZ, layer: 3, pos: 852
type: RSZ, layer: 3, pos: 1013
type: RSZ, layer: 3, pos: 51
type: RSZ, layer: 3, pos: 989
type: RSZ, layer: 3, pos: 833
type: RSZ, layer: 3, pos: 664
type: RSZ, layer: 3, pos: 82
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 628
type: RSZ, layer: 3, pos: 694
type: RSZ, layer: 3, pos: 53
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 349
type: RSZ, layer: 3, pos: 258
type: RSZ, layer: 3, pos: 428
type: RSZ, layer: 3, pos: 333
type: RSZ, layer: 3, pos: 105
type: RSZ, layer: 3, pos: 971
type: RSZ, layer: 3, pos: 365
type: RSZ, layer: 3, pos: 202
type: RSZ, layer: 3, pos: 266
type: RSZ, layer: 3, pos: 651
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 595
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 681
type: RSZ, layer: 3, pos: 683
type: RSZ, layer: 3, pos: 645
type: RSZ, layer: 3, pos: 421
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 296
type: RSZ, layer: 3, pos: 845
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 280
type: RSZ, layer: 3, pos: 1006
type: RSZ, layer: 3, pos: 1007
type: RSZ, layer: 3, pos: 648
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 970
type: RSZ, layer: 3, pos: 260
type: RSZ, layer: 3, pos: 857
type: RSZ, layer: 3, pos: 341
type: RSZ, layer: 3, pos: 320
type: RSZ, layer: 3, pos: 55
type: RSZ, layer: 3, pos: 973
type: RSZ, layer: 3, pos: 301
type: RSZ, layer: 3, pos: 328
type: RSZ, layer: 3, pos: 623
type: RSZ, layer: 3, pos: 78
type: RSZ, layer: 3, pos: 650
type: RSZ, layer: 3, pos: 1011
type: RSZ, layer: 3, pos: 321
type: RSZ, layer: 3, pos: 598
type: RSZ, layer: 3, pos: 127
type: RSZ, layer: 3, pos: 207
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 597
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 649
type: RSZ, layer: 3, pos: 304
type: RSZ, layer: 3, pos: 596
type: RSZ, layer: 3, pos: 682
type: RSZ, layer: 3, pos: 701
type: RSZ, layer: 3, pos: 86
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 247
type: RSZ, layer: 3, pos: 109
type: RSZ, layer: 3, pos: 853
type: RSZ, layer: 3, pos: 1002
type: RSZ, layer: 3, pos: 261
type: RSZ, layer: 3, pos: 672
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 862
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 855
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 687
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 593
type: RSZ, layer: 3, pos: 243
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 205
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 1008
type: RSZ, layer: 3, pos: 630
type: RSZ, layer: 3, pos: 126
type: RSZ, layer: 3, pos: 602
type: RSZ, layer: 3, pos: 215
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 589
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 60
type: RSZ, layer: 3, pos: 367
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 835
type: RSZ, layer: 3, pos: 642
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 586
type: RSZ, layer: 3, pos: 609
type: RSZ, layer: 3, pos: 352
type: RSZ, layer: 3, pos: 263
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 696
type: RSZ, layer: 3, pos: 590
type: RSZ, layer: 3, pos: 1022
type: RSZ, layer: 3, pos: 587
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 288
type: RSZ, layer: 3, pos: 842
type: RSZ, layer: 3, pos: 327
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 836
type: RSZ, layer: 3, pos: 990
type: RSZ, layer: 3, pos: 269
type: RSZ, layer: 3, pos: 52
type: RSZ, layer: 3, pos: 981
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 264
type: RSZ, layer: 3, pos: 631
type: RSZ, layer: 3, pos: 636
type: RSZ, layer: 3, pos: 256
type: RSZ, layer: 3, pos: 966
type: RSZ, layer: 3, pos: 246
type: RSZ, layer: 3, pos: 873
type: RSZ, layer: 3, pos: 968
type: RSZ, layer: 3, pos: 594
type: RSZ, layer: 3, pos: 413
type: RSZ, layer: 3, pos: 622
type: RSZ, layer: 3, pos: 119
type: RSZ, layer: 3, pos: 1016
type: RSZ, layer: 3, pos: 653
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 639
type: RSZ, layer: 3, pos: 585
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 987
type: RSZ, layer: 3, pos: 606
type: RSZ, layer: 3, pos: 979
type: RSZ, layer: 3, pos: 405
type: RSZ, layer: 3, pos: 998
type: RSZ, layer: 3, pos: 641
type: RSZ, layer: 3, pos: 257
type: RSZ, layer: 3, pos: 267
type: RSZ, layer: 3, pos: 217
type: RSZ, layer: 3, pos: 680
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 618
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 617
type: RSZ, layer: 3, pos: 1001
type: RSZ, layer: 3, pos: 834
type: RSZ, layer: 3, pos: 97
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 685
type: RSZ, layer: 3, pos: 125
type: RSZ, layer: 3, pos: 983
type: RSZ, layer: 3, pos: 591
type: RSZ, layer: 3, pos: 368
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 87
type: RSZ, layer: 3, pos: 607
type: RSZ, layer: 3, pos: 88
type: RSZ, layer: 3, pos: 322
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 960
type: RSZ, layer: 3, pos: 691
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 861
type: RSZ, layer: 3, pos: 89
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 74
type: RSZ, layer: 3, pos: 629
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 262
type: RSZ, layer: 3, pos: 112
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 659
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 209
type: RSZ, layer: 3, pos: 601
type: RSZ, layer: 3, pos: 588
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 580
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 868

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1007543, upper bound: 14.0765832
time: 52.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.5230048, upper bound: 14.0699769
time: 57.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.5578117, -0.2815094, -37.5578117, -0.2815094, -37.2763023, 37.2763023
1: -17.6220856, 10.4812832, -17.6220856, 10.4812832, -28.1033688, 28.1033688
2: -14.4312439, 10.0883579, -14.4312439, 10.0883579, -24.5196018, 24.5196018
3: -14.9140787, 14.0766926, -14.9140787, 14.0766926, -28.9907722, 28.9907722
4: -15.1334782, 14.7552061, -15.1334782, 14.7552061, -29.8886833, 29.8886833
5: -14.2266178, 15.1941137, -14.2266178, 15.1941137, -29.4207306, 29.4207306
6: -20.8351784, 10.3227959, -20.8351784, 10.3227959, -31.1579742, 31.1579742
7: -17.3737793, 16.5286179, -17.3737793, 16.5286179, -33.4994736, 33.4994774
8: -16.3228550, 19.1706161, -16.3228550, 19.1706161, -35.4375305, 35.4410057
9: -15.1635761, 13.7782021, -15.1635761, 13.7782021, -28.7364159, 28.7408867
10: -23.5488243, 17.2546597, -23.5488243, 17.2546597, -40.8034821, 40.8034821
11: -26.2301464, 10.3602743, -26.2301464, 10.3602743, -36.5904198, 36.5904198
12: -24.2345219, 12.1104794, -24.2345219, 12.1104794, -36.3450012, 36.3450012
13: -22.1990852, 18.4545364, -22.1990852, 18.4545364, -40.6536217, 40.6536217
14: -47.8707161, -0.4218502, -47.8707161, -0.4218502, -47.2374115, 47.2735443
15: -19.6529541, 10.3302298, -19.6529541, 10.3302298, -29.9831848, 29.9831848
16: -24.9793587, 13.2617989, -24.9793587, 13.2617989, -37.7672577, 37.7779312
17: -43.9613876, 12.4236822, -43.9613876, 12.4236822, -55.0495605, 55.0794220
18: -20.4577484, 12.4874125, -20.4577484, 12.4874125, -32.9451599, 32.9451599
19: -17.9183464, 4.2855158, -17.9183464, 4.2855158, -22.2038612, 22.2038612
20: -15.2772102, 8.4781389, -15.2772102, 8.4781389, -23.7553482, 23.7553482
21: -25.8919182, 3.7857480, -25.8919182, 3.7857480, -29.6776657, 29.6776657
22: -32.9530487, -0.8713036, -32.9530487, -0.8713036, -30.6945572, 30.6722393
23: -17.9241028, 8.9438848, -17.9241028, 8.9438848, -26.8679886, 26.8679886
24: -25.2896576, 7.3427725, -25.2896576, 7.3427725, -31.1230087, 31.1055393
25: -18.3243713, 10.8286724, -18.3243713, 10.8286724, -29.1530437, 29.1530437
26: -23.7112923, 14.9050426, -23.7112923, 14.9050426, -38.6163330, 38.6163330
27: -26.2949791, 6.6883535, -26.2949791, 6.6883535, -31.9780655, 31.9560127
28: -17.3191872, 10.6421642, -17.3191872, 10.6421642, -27.7717361, 27.7725029
29: -40.1622734, -5.2830868, -40.1622734, -5.2830868, -33.9048119, 33.8878403
30: -20.8790817, 12.3533125, -20.8790817, 12.3533125, -33.2323952, 33.2323952
31: -23.7122688, 7.0026283, -23.7122688, 7.0026283, -30.7148972, 30.7148972
32: -27.6387272, 4.3835154, -27.6387272, 4.3835154, -31.0836258, 31.0776768
33: -30.5414295, 14.6008110, -30.5414295, 14.6008110, -44.2403412, 44.2415161
34: -25.9748363, 9.9279261, -25.9748363, 9.9279261, -35.9027634, 35.9027634
35: -27.7439537, 10.9692993, -27.7439537, 10.9692993, -38.3017387, 38.3050423
36: -27.1807117, 10.9039993, -27.1807117, 10.9039993, -37.6316528, 37.6286354
37: -37.2227592, 9.6494160, -37.2227592, 9.6494160, -45.5469818, 45.5436707
38: -29.7080078, 13.9970264, -29.7080078, 13.9970264, -43.7050323, 43.7050323
39: -38.4577255, 11.6211214, -38.4577255, 11.6211214, -49.4315796, 49.4284668
40: -30.4051437, 9.7967424, -30.4051437, 9.7967424, -38.5361328, 38.5357323
41: -22.3953133, 9.5435400, -22.3953133, 9.5435400, -31.9364777, 31.9388542
42: -16.3960018, 7.5732532, -16.3960018, 7.5732532, -23.6196270, 23.6281357

Time for backsubstitution: 0.89 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 868
type: RSZ, layer: 3, pos: 292
type: RSZ, layer: 3, pos: 236
type: RSZ, layer: 3, pos: 228
type: RSZ, layer: 3, pos: 355
type: RSZ, layer: 3, pos: 357
type: RSZ, layer: 3, pos: 284
type: RSZ, layer: 3, pos: 997
type: RSZ, layer: 3, pos: 724
type: RSZ, layer: 3, pos: 380
type: RSZ, layer: 3, pos: 363
type: RSZ, layer: 3, pos: 892
type: RSZ, layer: 3, pos: 375
type: RSZ, layer: 3, pos: 353
type: RSZ, layer: 3, pos: 377
type: RSZ, layer: 3, pos: 293
type: RSZ, layer: 3, pos: 887
type: RSZ, layer: 3, pos: 348
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 869
type: RSZ, layer: 3, pos: 316
type: RSZ, layer: 3, pos: 875
type: RSZ, layer: 3, pos: 378
type: RSZ, layer: 3, pos: 289
type: RSZ, layer: 3, pos: 369
type: RSZ, layer: 3, pos: 893
type: RSZ, layer: 3, pos: 351
type: RSZ, layer: 3, pos: 991
type: RSZ, layer: 3, pos: 999
type: RSZ, layer: 3, pos: 988
type: RSZ, layer: 3, pos: 305
type: RSZ, layer: 3, pos: 871
type: RSZ, layer: 3, pos: 881
type: RSZ, layer: 3, pos: 996
type: RSZ, layer: 3, pos: 876
type: RSZ, layer: 3, pos: 383
type: RSZ, layer: 3, pos: 299
type: RSZ, layer: 3, pos: 361
type: RSZ, layer: 3, pos: 877
type: RSZ, layer: 3, pos: 1015
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 381
type: RSZ, layer: 3, pos: 291
type: RSZ, layer: 3, pos: 306
type: RSZ, layer: 3, pos: 993
type: RSZ, layer: 3, pos: 382
type: RSZ, layer: 3, pos: 851
type: RSZ, layer: 3, pos: 334
type: RSZ, layer: 3, pos: 843
type: RSZ, layer: 3, pos: 865
type: RSZ, layer: 3, pos: 314
type: RSZ, layer: 3, pos: 282
type: RSZ, layer: 3, pos: 858
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 859
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 889
type: RSZ, layer: 3, pos: 863
type: RSZ, layer: 3, pos: 895
type: RSZ, layer: 3, pos: 1009
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 274
type: RSZ, layer: 3, pos: 346
type: RSZ, layer: 3, pos: 283
type: RSZ, layer: 3, pos: 849
type: RSZ, layer: 3, pos: 1004
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 271
type: RSZ, layer: 3, pos: 379
type: RSZ, layer: 3, pos: 1023
type: RSZ, layer: 3, pos: 972
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 335
type: RSZ, layer: 3, pos: 850
type: RSZ, layer: 3, pos: 986
type: RSZ, layer: 3, pos: 279
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 700
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 980
type: RSZ, layer: 3, pos: 1005
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 884
type: RSZ, layer: 3, pos: 340
type: RSZ, layer: 3, pos: 1003
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 343
type: RSZ, layer: 3, pos: 846
type: RSZ, layer: 3, pos: 684
type: RSZ, layer: 3, pos: 689
type: RSZ, layer: 3, pos: 885
type: RSZ, layer: 3, pos: 315
type: RSZ, layer: 3, pos: 1020
type: RSZ, layer: 3, pos: 978
type: RSZ, layer: 3, pos: 995
type: RSZ, layer: 3, pos: 300
type: RSZ, layer: 3, pos: 331
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 319
type: RSZ, layer: 3, pos: 882
type: RSZ, layer: 3, pos: 364
type: RSZ, layer: 3, pos: 370
type: RSZ, layer: 3, pos: 84
type: RSZ, layer: 3, pos: 1021
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 673
type: RSZ, layer: 3, pos: 235
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 338
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 63
type: RSZ, layer: 3, pos: 699
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 58
type: RSZ, layer: 3, pos: 85
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 297
type: RSZ, layer: 3, pos: 239
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 339
type: RSZ, layer: 3, pos: 281
type: RSZ, layer: 3, pos: 69
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1019
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 344
type: RSZ, layer: 3, pos: 68
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 270
type: RSZ, layer: 3, pos: 883
type: RSZ, layer: 3, pos: 358
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 372
type: RSZ, layer: 3, pos: 666
type: RSZ, layer: 3, pos: 974
type: RSZ, layer: 3, pos: 667
type: RSZ, layer: 3, pos: 644
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 656
type: RSZ, layer: 3, pos: 870
type: RSZ, layer: 3, pos: 690
type: RSZ, layer: 3, pos: 273
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 336
type: RSZ, layer: 3, pos: 647
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 860
type: RSZ, layer: 3, pos: 841
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 318
type: RSZ, layer: 3, pos: 977
type: RSZ, layer: 3, pos: 61
type: RSZ, layer: 3, pos: 879
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 345
type: RSZ, layer: 3, pos: 663
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 110
type: RSZ, layer: 3, pos: 1017
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 102
type: RSZ, layer: 3, pos: 967
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 688
type: RSZ, layer: 3, pos: 50
type: RSZ, layer: 3, pos: 214
type: RSZ, layer: 3, pos: 867
type: RSZ, layer: 3, pos: 1014
type: RSZ, layer: 3, pos: 652
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 354
type: RSZ, layer: 3, pos: 114
type: RSZ, layer: 3, pos: 1012
type: RSZ, layer: 3, pos: 1018
type: RSZ, layer: 3, pos: 329
type: RSZ, layer: 3, pos: 844
type: RSZ, layer: 3, pos: 703
type: RSZ, layer: 3, pos: 326
type: RSZ, layer: 3, pos: 201
type: RSZ, layer: 3, pos: 376
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 1010
type: RSZ, layer: 3, pos: 259
type: RSZ, layer: 3, pos: 371
type: RSZ, layer: 3, pos: 874
type: RSZ, layer: 3, pos: 57
type: RSZ, layer: 3, pos: 272
type: RSZ, layer: 3, pos: 347
type: RSZ, layer: 3, pos: 692
type: RSZ, layer: 3, pos: 646
type: RSZ, layer: 3, pos: 275
type: RSZ, layer: 3, pos: 56
type: RSZ, layer: 3, pos: 203
type: RSZ, layer: 3, pos: 657
type: RSZ, layer: 3, pos: 420
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 276
type: RSZ, layer: 3, pos: 113
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 265
type: RSZ, layer: 3, pos: 419
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 210
type: RSZ, layer: 3, pos: 324
type: RSZ, layer: 3, pos: 123
type: RSZ, layer: 3, pos: 285
type: RSZ, layer: 3, pos: 965
type: RSZ, layer: 3, pos: 303
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 643
type: RSZ, layer: 3, pos: 847
type: RSZ, layer: 3, pos: 94
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 985
type: RSZ, layer: 3, pos: 658
type: RSZ, layer: 3, pos: 362
type: RSZ, layer: 3, pos: 665
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 674
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 360
type: RSZ, layer: 3, pos: 702
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 54
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 982
type: RSZ, layer: 3, pos: 866
type: RSZ, layer: 3, pos: 852
type: RSZ, layer: 3, pos: 1013
type: RSZ, layer: 3, pos: 51
type: RSZ, layer: 3, pos: 989
type: RSZ, layer: 3, pos: 833
type: RSZ, layer: 3, pos: 664
type: RSZ, layer: 3, pos: 82
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 628
type: RSZ, layer: 3, pos: 694
type: RSZ, layer: 3, pos: 53
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 349
type: RSZ, layer: 3, pos: 258
type: RSZ, layer: 3, pos: 428
type: RSZ, layer: 3, pos: 333
type: RSZ, layer: 3, pos: 105
type: RSZ, layer: 3, pos: 971
type: RSZ, layer: 3, pos: 365
type: RSZ, layer: 3, pos: 202
type: RSZ, layer: 3, pos: 266
type: RSZ, layer: 3, pos: 651
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 595
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 681
type: RSZ, layer: 3, pos: 683
type: RSZ, layer: 3, pos: 645
type: RSZ, layer: 3, pos: 421
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 296
type: RSZ, layer: 3, pos: 845
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 280
type: RSZ, layer: 3, pos: 1006
type: RSZ, layer: 3, pos: 1007
type: RSZ, layer: 3, pos: 648
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 970
type: RSZ, layer: 3, pos: 260
type: RSZ, layer: 3, pos: 857
type: RSZ, layer: 3, pos: 341
type: RSZ, layer: 3, pos: 320
type: RSZ, layer: 3, pos: 55
type: RSZ, layer: 3, pos: 973
type: RSZ, layer: 3, pos: 301
type: RSZ, layer: 3, pos: 328
type: RSZ, layer: 3, pos: 623
type: RSZ, layer: 3, pos: 78
type: RSZ, layer: 3, pos: 650
type: RSZ, layer: 3, pos: 1011
type: RSZ, layer: 3, pos: 321
type: RSZ, layer: 3, pos: 598
type: RSZ, layer: 3, pos: 127
type: RSZ, layer: 3, pos: 207
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 597
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 649
type: RSZ, layer: 3, pos: 304
type: RSZ, layer: 3, pos: 596
type: RSZ, layer: 3, pos: 682
type: RSZ, layer: 3, pos: 701
type: RSZ, layer: 3, pos: 86
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 247
type: RSZ, layer: 3, pos: 109
type: RSZ, layer: 3, pos: 853
type: RSZ, layer: 3, pos: 1002
type: RSZ, layer: 3, pos: 261
type: RSZ, layer: 3, pos: 672
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 862
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 855
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 687
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 593
type: RSZ, layer: 3, pos: 243
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 205
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 1008
type: RSZ, layer: 3, pos: 630
type: RSZ, layer: 3, pos: 126
type: RSZ, layer: 3, pos: 602
type: RSZ, layer: 3, pos: 215
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 589
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 60
type: RSZ, layer: 3, pos: 367
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 835
type: RSZ, layer: 3, pos: 642
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 586
type: RSZ, layer: 3, pos: 609
type: RSZ, layer: 3, pos: 352
type: RSZ, layer: 3, pos: 263
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 696
type: RSZ, layer: 3, pos: 590
type: RSZ, layer: 3, pos: 1022
type: RSZ, layer: 3, pos: 587
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 288
type: RSZ, layer: 3, pos: 842
type: RSZ, layer: 3, pos: 327
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 836
type: RSZ, layer: 3, pos: 990
type: RSZ, layer: 3, pos: 269
type: RSZ, layer: 3, pos: 52
type: RSZ, layer: 3, pos: 981
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 264
type: RSZ, layer: 3, pos: 631
type: RSZ, layer: 3, pos: 636
type: RSZ, layer: 3, pos: 256
type: RSZ, layer: 3, pos: 966
type: RSZ, layer: 3, pos: 246
type: RSZ, layer: 3, pos: 873
type: RSZ, layer: 3, pos: 968
type: RSZ, layer: 3, pos: 594
type: RSZ, layer: 3, pos: 413
type: RSZ, layer: 3, pos: 622
type: RSZ, layer: 3, pos: 119
type: RSZ, layer: 3, pos: 1016
type: RSZ, layer: 3, pos: 653
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 639
type: RSZ, layer: 3, pos: 585
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 987
type: RSZ, layer: 3, pos: 606
type: RSZ, layer: 3, pos: 979
type: RSZ, layer: 3, pos: 405
type: RSZ, layer: 3, pos: 998
type: RSZ, layer: 3, pos: 641
type: RSZ, layer: 3, pos: 257
type: RSZ, layer: 3, pos: 267
type: RSZ, layer: 3, pos: 217
type: RSZ, layer: 3, pos: 680
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 618
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 617
type: RSZ, layer: 3, pos: 1001
type: RSZ, layer: 3, pos: 834
type: RSZ, layer: 3, pos: 97
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 685
type: RSZ, layer: 3, pos: 125
type: RSZ, layer: 3, pos: 983
type: RSZ, layer: 3, pos: 591
type: RSZ, layer: 3, pos: 368
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 87
type: RSZ, layer: 3, pos: 607
type: RSZ, layer: 3, pos: 88
type: RSZ, layer: 3, pos: 322
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 960
type: RSZ, layer: 3, pos: 691
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 861
type: RSZ, layer: 3, pos: 89
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 74
type: RSZ, layer: 3, pos: 629
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 262
type: RSZ, layer: 3, pos: 112
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 659
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 209
type: RSZ, layer: 3, pos: 601
type: RSZ, layer: 3, pos: 588
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 580
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 3, pos: 868

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1064588, upper bound: 14.1091648
time: 60.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.5099357, upper bound: 14.0843435
time: 61.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.5578117, -0.2815094, -37.5578117, -0.2815094, -37.2763023, 37.2763023
1: -17.6220856, 10.4812832, -17.6220856, 10.4812832, -28.1033688, 28.1033688
2: -14.4312439, 10.0883579, -14.4312439, 10.0883579, -24.5196018, 24.5196018
3: -14.9140787, 14.0766926, -14.9140787, 14.0766926, -28.9907722, 28.9907722
4: -15.1334782, 14.7552061, -15.1334782, 14.7552061, -29.8886833, 29.8886833
5: -14.2266178, 15.1941137, -14.2266178, 15.1941137, -29.4207306, 29.4207306
6: -20.8351784, 10.3227959, -20.8351784, 10.3227959, -31.1579742, 31.1579742
7: -17.3737793, 16.5286179, -17.3737793, 16.5286179, -33.5000076, 33.4990654
8: -16.3228550, 19.1706161, -16.3228550, 19.1706161, -35.4393158, 35.4391136
9: -15.1635761, 13.7782021, -15.1635761, 13.7782021, -28.7375908, 28.7397518
10: -23.5488243, 17.2546597, -23.5488243, 17.2546597, -40.8034821, 40.8034821
11: -26.2301464, 10.3602743, -26.2301464, 10.3602743, -36.5904198, 36.5904198
12: -24.2345219, 12.1104794, -24.2345219, 12.1104794, -36.3450012, 36.3450012
13: -22.1990852, 18.4545364, -22.1990852, 18.4545364, -40.6536217, 40.6536217
14: -47.8707161, -0.4218502, -47.8707161, -0.4218502, -47.2372589, 47.2739334
15: -19.6529541, 10.3302298, -19.6529541, 10.3302298, -29.9831848, 29.9831848
16: -24.9793587, 13.2617989, -24.9793587, 13.2617989, -37.7722549, 37.7729187
17: -43.9613876, 12.4236822, -43.9613876, 12.4236822, -55.0487976, 55.0803680
18: -20.4577484, 12.4874125, -20.4577484, 12.4874125, -32.9451599, 32.9451599
19: -17.9183464, 4.2855158, -17.9183464, 4.2855158, -22.2038612, 22.2038612
20: -15.2772102, 8.4781389, -15.2772102, 8.4781389, -23.7553482, 23.7553482
21: -25.8919182, 3.7857480, -25.8919182, 3.7857480, -29.6776657, 29.6776657
22: -32.9530487, -0.8713036, -32.9530487, -0.8713036, -30.6855545, 30.6818657
23: -17.9241028, 8.9438848, -17.9241028, 8.9438848, -26.8679886, 26.8679886
24: -25.2896576, 7.3427725, -25.2896576, 7.3427725, -31.1177902, 31.1112137
25: -18.3243713, 10.8286724, -18.3243713, 10.8286724, -29.1530437, 29.1530437
26: -23.7112923, 14.9050426, -23.7112923, 14.9050426, -38.6163330, 38.6163330
27: -26.2949791, 6.6883535, -26.2949791, 6.6883535, -31.9774551, 31.9566689
28: -17.3191872, 10.6421642, -17.3191872, 10.6421642, -27.7714310, 27.7728004
29: -40.1622734, -5.2830868, -40.1622734, -5.2830868, -33.8952599, 33.8976555
30: -20.8790817, 12.3533125, -20.8790817, 12.3533125, -33.2323952, 33.2323952
31: -23.7122688, 7.0026283, -23.7122688, 7.0026283, -30.7148972, 30.7148972
32: -27.6387272, 4.3835154, -27.6387272, 4.3835154, -31.0839081, 31.0774765
33: -30.5414295, 14.6008110, -30.5414295, 14.6008110, -44.2438583, 44.2380600
34: -25.9748363, 9.9279261, -25.9748363, 9.9279261, -35.9027634, 35.9027634
35: -27.7439537, 10.9692993, -27.7439537, 10.9692993, -38.3045921, 38.3021507
36: -27.1807117, 10.9039993, -27.1807117, 10.9039993, -37.6318665, 37.6284676
37: -37.2227592, 9.6494160, -37.2227592, 9.6494160, -45.5484543, 45.5422058
38: -29.7080078, 13.9970264, -29.7080078, 13.9970264, -43.7050323, 43.7050323
39: -38.4577255, 11.6211214, -38.4577255, 11.6211214, -49.4305496, 49.4295578
40: -30.4051437, 9.7967424, -30.4051437, 9.7967424, -38.5366821, 38.5352058
41: -22.3953133, 9.5435400, -22.3953133, 9.5435400, -31.9369202, 31.9388542
42: -16.3960018, 7.5732532, -16.3960018, 7.5732532, -23.6210308, 23.6267300

Time for backsubstitution: 0.93 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 868
type: RSZ, layer: 3, pos: 292
type: RSZ, layer: 3, pos: 236
type: RSZ, layer: 3, pos: 228
type: RSZ, layer: 3, pos: 355
type: RSZ, layer: 3, pos: 357
type: RSZ, layer: 3, pos: 284
type: RSZ, layer: 3, pos: 997
type: RSZ, layer: 3, pos: 724
type: RSZ, layer: 3, pos: 380
type: RSZ, layer: 3, pos: 363
type: RSZ, layer: 3, pos: 892
type: RSZ, layer: 3, pos: 375
type: RSZ, layer: 3, pos: 353
type: RSZ, layer: 3, pos: 377
type: RSZ, layer: 3, pos: 293
type: RSZ, layer: 3, pos: 887
type: RSZ, layer: 3, pos: 348
type: RSZ, layer: 3, pos: 311
type: RSZ, layer: 3, pos: 869
type: RSZ, layer: 3, pos: 316
type: RSZ, layer: 3, pos: 875
type: RSZ, layer: 3, pos: 378
type: RSZ, layer: 3, pos: 289
type: RSZ, layer: 3, pos: 369
type: RSZ, layer: 3, pos: 893
type: RSZ, layer: 3, pos: 351
type: RSZ, layer: 3, pos: 991
type: RSZ, layer: 3, pos: 999
type: RSZ, layer: 3, pos: 988
type: RSZ, layer: 3, pos: 305
type: RSZ, layer: 3, pos: 871
type: RSZ, layer: 3, pos: 881
type: RSZ, layer: 3, pos: 996
type: RSZ, layer: 3, pos: 876
type: RSZ, layer: 3, pos: 383
type: RSZ, layer: 3, pos: 299
type: RSZ, layer: 3, pos: 361
type: RSZ, layer: 3, pos: 877
type: RSZ, layer: 3, pos: 1015
type: RSZ, layer: 3, pos: 668
type: RSZ, layer: 3, pos: 381
type: RSZ, layer: 3, pos: 291
type: RSZ, layer: 3, pos: 306
type: RSZ, layer: 3, pos: 993
type: RSZ, layer: 3, pos: 382
type: RSZ, layer: 3, pos: 851
type: RSZ, layer: 3, pos: 334
type: RSZ, layer: 3, pos: 843
type: RSZ, layer: 3, pos: 865
type: RSZ, layer: 3, pos: 314
type: RSZ, layer: 3, pos: 282
type: RSZ, layer: 3, pos: 858
type: RSZ, layer: 3, pos: 695
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 859
type: RSZ, layer: 3, pos: 891
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 889
type: RSZ, layer: 3, pos: 863
type: RSZ, layer: 3, pos: 895
type: RSZ, layer: 3, pos: 1009
type: RSZ, layer: 3, pos: 317
type: RSZ, layer: 3, pos: 274
type: RSZ, layer: 3, pos: 346
type: RSZ, layer: 3, pos: 283
type: RSZ, layer: 3, pos: 849
type: RSZ, layer: 3, pos: 1004
type: RSZ, layer: 3, pos: 231
type: RSZ, layer: 3, pos: 271
type: RSZ, layer: 3, pos: 379
type: RSZ, layer: 3, pos: 1023
type: RSZ, layer: 3, pos: 972
type: RSZ, layer: 3, pos: 313
type: RSZ, layer: 3, pos: 335
type: RSZ, layer: 3, pos: 850
type: RSZ, layer: 3, pos: 986
type: RSZ, layer: 3, pos: 279
type: RSZ, layer: 3, pos: 655
type: RSZ, layer: 3, pos: 700
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 980
type: RSZ, layer: 3, pos: 1005
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 884
type: RSZ, layer: 3, pos: 340
type: RSZ, layer: 3, pos: 1003
type: RSZ, layer: 3, pos: 220
type: RSZ, layer: 3, pos: 343
type: RSZ, layer: 3, pos: 846
type: RSZ, layer: 3, pos: 684
type: RSZ, layer: 3, pos: 689
type: RSZ, layer: 3, pos: 885
type: RSZ, layer: 3, pos: 315
type: RSZ, layer: 3, pos: 1020
type: RSZ, layer: 3, pos: 978
type: RSZ, layer: 3, pos: 995
type: RSZ, layer: 3, pos: 300
type: RSZ, layer: 3, pos: 331
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 319
type: RSZ, layer: 3, pos: 882
type: RSZ, layer: 3, pos: 364
type: RSZ, layer: 3, pos: 370
type: RSZ, layer: 3, pos: 84
type: RSZ, layer: 3, pos: 1021
type: RSZ, layer: 3, pos: 894
type: RSZ, layer: 3, pos: 673
type: RSZ, layer: 3, pos: 235
type: RSZ, layer: 3, pos: 223
type: RSZ, layer: 3, pos: 338
type: RSZ, layer: 3, pos: 62
type: RSZ, layer: 3, pos: 63
type: RSZ, layer: 3, pos: 699
type: RSZ, layer: 3, pos: 963
type: RSZ, layer: 3, pos: 58
type: RSZ, layer: 3, pos: 85
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 39
type: RSZ, layer: 3, pos: 297
type: RSZ, layer: 3, pos: 239
type: RSZ, layer: 3, pos: 255
type: RSZ, layer: 3, pos: 339
type: RSZ, layer: 3, pos: 281
type: RSZ, layer: 3, pos: 69
type: RSZ, layer: 3, pos: 698
type: RSZ, layer: 3, pos: 1019
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 344
type: RSZ, layer: 3, pos: 68
type: RSZ, layer: 3, pos: 234
type: RSZ, layer: 3, pos: 270
type: RSZ, layer: 3, pos: 883
type: RSZ, layer: 3, pos: 358
type: RSZ, layer: 3, pos: 225
type: RSZ, layer: 3, pos: 372
type: RSZ, layer: 3, pos: 666
type: RSZ, layer: 3, pos: 974
type: RSZ, layer: 3, pos: 667
type: RSZ, layer: 3, pos: 644
type: RSZ, layer: 3, pos: 204
type: RSZ, layer: 3, pos: 124
type: RSZ, layer: 3, pos: 697
type: RSZ, layer: 3, pos: 656
type: RSZ, layer: 3, pos: 870
type: RSZ, layer: 3, pos: 690
type: RSZ, layer: 3, pos: 273
type: RSZ, layer: 3, pos: 975
type: RSZ, layer: 3, pos: 15
type: RSZ, layer: 3, pos: 336
type: RSZ, layer: 3, pos: 647
type: RSZ, layer: 3, pos: 242
type: RSZ, layer: 3, pos: 860
type: RSZ, layer: 3, pos: 841
type: RSZ, layer: 3, pos: 238
type: RSZ, layer: 3, pos: 318
type: RSZ, layer: 3, pos: 977
type: RSZ, layer: 3, pos: 61
type: RSZ, layer: 3, pos: 879
type: RSZ, layer: 3, pos: 47
type: RSZ, layer: 3, pos: 345
type: RSZ, layer: 3, pos: 663
type: RSZ, layer: 3, pos: 241
type: RSZ, layer: 3, pos: 110
type: RSZ, layer: 3, pos: 1017
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 253
type: RSZ, layer: 3, pos: 102
type: RSZ, layer: 3, pos: 967
type: RSZ, layer: 3, pos: 41
type: RSZ, layer: 3, pos: 688
type: RSZ, layer: 3, pos: 50
type: RSZ, layer: 3, pos: 214
type: RSZ, layer: 3, pos: 867
type: RSZ, layer: 3, pos: 1014
type: RSZ, layer: 3, pos: 652
type: RSZ, layer: 3, pos: 337
type: RSZ, layer: 3, pos: 354
type: RSZ, layer: 3, pos: 114
type: RSZ, layer: 3, pos: 1012
type: RSZ, layer: 3, pos: 1018
type: RSZ, layer: 3, pos: 329
type: RSZ, layer: 3, pos: 844
type: RSZ, layer: 3, pos: 703
type: RSZ, layer: 3, pos: 326
type: RSZ, layer: 3, pos: 201
type: RSZ, layer: 3, pos: 376
type: RSZ, layer: 3, pos: 66
type: RSZ, layer: 3, pos: 1010
type: RSZ, layer: 3, pos: 259
type: RSZ, layer: 3, pos: 371
type: RSZ, layer: 3, pos: 874
type: RSZ, layer: 3, pos: 57
type: RSZ, layer: 3, pos: 272
type: RSZ, layer: 3, pos: 347
type: RSZ, layer: 3, pos: 692
type: RSZ, layer: 3, pos: 646
type: RSZ, layer: 3, pos: 275
type: RSZ, layer: 3, pos: 56
type: RSZ, layer: 3, pos: 203
type: RSZ, layer: 3, pos: 657
type: RSZ, layer: 3, pos: 420
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 276
type: RSZ, layer: 3, pos: 113
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 265
type: RSZ, layer: 3, pos: 419
type: RSZ, layer: 3, pos: 71
type: RSZ, layer: 3, pos: 210
type: RSZ, layer: 3, pos: 324
type: RSZ, layer: 3, pos: 123
type: RSZ, layer: 3, pos: 285
type: RSZ, layer: 3, pos: 965
type: RSZ, layer: 3, pos: 303
type: RSZ, layer: 3, pos: 222
type: RSZ, layer: 3, pos: 643
type: RSZ, layer: 3, pos: 847
type: RSZ, layer: 3, pos: 94
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 73
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 985
type: RSZ, layer: 3, pos: 658
type: RSZ, layer: 3, pos: 362
type: RSZ, layer: 3, pos: 665
type: RSZ, layer: 3, pos: 675
type: RSZ, layer: 3, pos: 674
type: RSZ, layer: 3, pos: 249
type: RSZ, layer: 3, pos: 360
type: RSZ, layer: 3, pos: 702
type: RSZ, layer: 3, pos: 252
type: RSZ, layer: 3, pos: 54
type: RSZ, layer: 3, pos: 103
type: RSZ, layer: 3, pos: 982
type: RSZ, layer: 3, pos: 866
type: RSZ, layer: 3, pos: 852
type: RSZ, layer: 3, pos: 1013
type: RSZ, layer: 3, pos: 51
type: RSZ, layer: 3, pos: 989
type: RSZ, layer: 3, pos: 833
type: RSZ, layer: 3, pos: 664
type: RSZ, layer: 3, pos: 82
type: RSZ, layer: 3, pos: 404
type: RSZ, layer: 3, pos: 628
type: RSZ, layer: 3, pos: 694
type: RSZ, layer: 3, pos: 53
type: RSZ, layer: 3, pos: 59
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 349
type: RSZ, layer: 3, pos: 258
type: RSZ, layer: 3, pos: 428
type: RSZ, layer: 3, pos: 333
type: RSZ, layer: 3, pos: 105
type: RSZ, layer: 3, pos: 971
type: RSZ, layer: 3, pos: 365
type: RSZ, layer: 3, pos: 202
type: RSZ, layer: 3, pos: 266
type: RSZ, layer: 3, pos: 651
type: RSZ, layer: 3, pos: 251
type: RSZ, layer: 3, pos: 595
type: RSZ, layer: 3, pos: 79
type: RSZ, layer: 3, pos: 681
type: RSZ, layer: 3, pos: 683
type: RSZ, layer: 3, pos: 645
type: RSZ, layer: 3, pos: 421
type: RSZ, layer: 3, pos: 677
type: RSZ, layer: 3, pos: 296
type: RSZ, layer: 3, pos: 845
type: RSZ, layer: 3, pos: 70
type: RSZ, layer: 3, pos: 280
type: RSZ, layer: 3, pos: 1006
type: RSZ, layer: 3, pos: 1007
type: RSZ, layer: 3, pos: 648
type: RSZ, layer: 3, pos: 29
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 970
type: RSZ, layer: 3, pos: 260
type: RSZ, layer: 3, pos: 857
type: RSZ, layer: 3, pos: 341
type: RSZ, layer: 3, pos: 320
type: RSZ, layer: 3, pos: 55
type: RSZ, layer: 3, pos: 973
type: RSZ, layer: 3, pos: 301
type: RSZ, layer: 3, pos: 328
type: RSZ, layer: 3, pos: 623
type: RSZ, layer: 3, pos: 78
type: RSZ, layer: 3, pos: 650
type: RSZ, layer: 3, pos: 1011
type: RSZ, layer: 3, pos: 321
type: RSZ, layer: 3, pos: 598
type: RSZ, layer: 3, pos: 127
type: RSZ, layer: 3, pos: 207
type: RSZ, layer: 3, pos: 72
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 597
type: RSZ, layer: 3, pos: 65
type: RSZ, layer: 3, pos: 649
type: RSZ, layer: 3, pos: 304
type: RSZ, layer: 3, pos: 596
type: RSZ, layer: 3, pos: 682
type: RSZ, layer: 3, pos: 701
type: RSZ, layer: 3, pos: 86
type: RSZ, layer: 3, pos: 122
type: RSZ, layer: 3, pos: 247
type: RSZ, layer: 3, pos: 109
type: RSZ, layer: 3, pos: 853
type: RSZ, layer: 3, pos: 1002
type: RSZ, layer: 3, pos: 261
type: RSZ, layer: 3, pos: 672
type: RSZ, layer: 3, pos: 111
type: RSZ, layer: 3, pos: 81
type: RSZ, layer: 3, pos: 862
type: RSZ, layer: 3, pos: 9
type: RSZ, layer: 3, pos: 855
type: RSZ, layer: 3, pos: 610
type: RSZ, layer: 3, pos: 687
type: RSZ, layer: 3, pos: 80
type: RSZ, layer: 3, pos: 206
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 593
type: RSZ, layer: 3, pos: 243
type: RSZ, layer: 3, pos: 34
type: RSZ, layer: 3, pos: 205
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 1008
type: RSZ, layer: 3, pos: 630
type: RSZ, layer: 3, pos: 126
type: RSZ, layer: 3, pos: 602
type: RSZ, layer: 3, pos: 215
type: RSZ, layer: 3, pos: 77
type: RSZ, layer: 3, pos: 589
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 669
type: RSZ, layer: 3, pos: 60
type: RSZ, layer: 3, pos: 367
type: RSZ, layer: 3, pos: 17
type: RSZ, layer: 3, pos: 835
type: RSZ, layer: 3, pos: 642
type: RSZ, layer: 3, pos: 725
type: RSZ, layer: 3, pos: 120
type: RSZ, layer: 3, pos: 586
type: RSZ, layer: 3, pos: 609
type: RSZ, layer: 3, pos: 352
type: RSZ, layer: 3, pos: 263
type: RSZ, layer: 3, pos: 969
type: RSZ, layer: 3, pos: 696
type: RSZ, layer: 3, pos: 590
type: RSZ, layer: 3, pos: 1022
type: RSZ, layer: 3, pos: 587
type: RSZ, layer: 3, pos: 230
type: RSZ, layer: 3, pos: 288
type: RSZ, layer: 3, pos: 842
type: RSZ, layer: 3, pos: 327
type: RSZ, layer: 3, pos: 104
type: RSZ, layer: 3, pos: 836
type: RSZ, layer: 3, pos: 990
type: RSZ, layer: 3, pos: 269
type: RSZ, layer: 3, pos: 52
type: RSZ, layer: 3, pos: 981
type: RSZ, layer: 3, pos: 16
type: RSZ, layer: 3, pos: 250
type: RSZ, layer: 3, pos: 233
type: RSZ, layer: 3, pos: 24
type: RSZ, layer: 3, pos: 264
type: RSZ, layer: 3, pos: 631
type: RSZ, layer: 3, pos: 636
type: RSZ, layer: 3, pos: 256
type: RSZ, layer: 3, pos: 966
type: RSZ, layer: 3, pos: 246
type: RSZ, layer: 3, pos: 873
type: RSZ, layer: 3, pos: 968
type: RSZ, layer: 3, pos: 594
type: RSZ, layer: 3, pos: 413
type: RSZ, layer: 3, pos: 622
type: RSZ, layer: 3, pos: 119
type: RSZ, layer: 3, pos: 1016
type: RSZ, layer: 3, pos: 653
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 639
type: RSZ, layer: 3, pos: 585
type: RSZ, layer: 3, pos: 49
type: RSZ, layer: 3, pos: 987
type: RSZ, layer: 3, pos: 606
type: RSZ, layer: 3, pos: 979
type: RSZ, layer: 3, pos: 405
type: RSZ, layer: 3, pos: 998
type: RSZ, layer: 3, pos: 641
type: RSZ, layer: 3, pos: 257
type: RSZ, layer: 3, pos: 267
type: RSZ, layer: 3, pos: 217
type: RSZ, layer: 3, pos: 680
type: RSZ, layer: 3, pos: 654
type: RSZ, layer: 3, pos: 618
type: RSZ, layer: 3, pos: 599
type: RSZ, layer: 3, pos: 617
type: RSZ, layer: 3, pos: 1001
type: RSZ, layer: 3, pos: 834
type: RSZ, layer: 3, pos: 97
type: RSZ, layer: 3, pos: 23
type: RSZ, layer: 3, pos: 685
type: RSZ, layer: 3, pos: 125
type: RSZ, layer: 3, pos: 983
type: RSZ, layer: 3, pos: 591
type: RSZ, layer: 3, pos: 368
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 87
type: RSZ, layer: 3, pos: 607
type: RSZ, layer: 3, pos: 88
type: RSZ, layer: 3, pos: 322
type: RSZ, layer: 3, pos: 627
type: RSZ, layer: 3, pos: 43
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 960
type: RSZ, layer: 3, pos: 691
type: RSZ, layer: 3, pos: 96
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 861
type: RSZ, layer: 3, pos: 89
type: RSZ, layer: 3, pos: 32
type: RSZ, layer: 3, pos: 74
type: RSZ, layer: 3, pos: 629
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 262
type: RSZ, layer: 3, pos: 112
type: RSZ, layer: 3, pos: 254
type: RSZ, layer: 3, pos: 659
type: RSZ, layer: 3, pos: 325
type: RSZ, layer: 3, pos: 31
type: RSZ, layer: 3, pos: 209
type: RSZ, layer: 3, pos: 601
type: RSZ, layer: 3, pos: 588
type: RSZ, layer: 3, pos: 626
type: RSZ, layer: 3, pos: 580
type: RSZ, layer: 3, pos: 121

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 3, pos: 868

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.1828287, upper bound: 13.9933462
time: 68.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -14.6311161, upper bound: 13.9782079
time: 66.25 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 135.87 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 135.87
Output dim: 4, lower bound: -13.9782079, upper bound: 14.6311161
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 135.87
Output dim: 4, lower bound: -13.9933462, upper bound: 14.1828287
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 135.87
Output dim: 4, lower bound: -14.0843435, upper bound: 14.5099357
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 135.87
Output dim: 4, lower bound: -14.1091648, upper bound: 14.1064588
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 135.87
Output dim: 4, lower bound: -14.0699769, upper bound: 14.5230048
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 135.87
Output dim: 4, lower bound: -14.0765832, upper bound: 14.1007543
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 135.87
Output dim: 4, lower bound: -14.1549993, upper bound: 14.3802180
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 135.87
Output dim: 4, lower bound: -14.1669745, upper bound: 14.0018900
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 135.87
Output dim: 4, lower bound: -14.0018900, upper bound: 14.1669745
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 135.87
Output dim: 4, lower bound: -14.3802180, upper bound: 14.1549993
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 135.87
Output dim: 4, lower bound: -14.1007543, upper bound: 14.0765832
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 135.87
Output dim: 4, lower bound: -14.5230048, upper bound: 14.0699769
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 135.87
Output dim: 4, lower bound: -14.1064588, upper bound: 14.1091648
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 135.87
Output dim: 4, lower bound: -14.5099357, upper bound: 14.0843435
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 135.87
Output dim: 4, lower bound: -14.1828287, upper bound: 13.9933462
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 135.87
Output dim: 4, lower bound: -14.6311161, upper bound: 13.9782079

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 65.83 + 1797.66 = 1863.49 seconds
