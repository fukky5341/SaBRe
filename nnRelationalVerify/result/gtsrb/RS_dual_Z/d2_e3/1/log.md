## Execution arguments:
Dataset: Dataset.GTSRB
Network: onnx/gtsrb_small_cnn.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 3600 seconds
Split limit: 100


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498)
1: (-11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070)
2: (-9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425)
3: (-9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4056396, 38.4056396)
4: (-16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9559555, 41.9559555)
5: (-7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1119690, 36.1119690)
6: (-38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812)
7: (-11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.6014786, 38.6014748)
8: (-21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7545319, 50.7545319)
9: (-13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186)
10: (-22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662)
11: (-23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790)
12: (-44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2175751, 45.2175751)
13: (-37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5576782, 59.5576782)
14: (-64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507)
15: (-21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186)
16: (-23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021)
17: (-58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0626373, 56.0626373)
18: (-35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650)
19: (-26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775)
20: (-21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115)
21: (-27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035)
22: (-32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925)
23: (-24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843)
24: (-30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705)
25: (-28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095)
26: (-41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693)
27: (-26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029)
28: (-25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052)
29: (-27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4229507, 38.4229507)
30: (-26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750)
31: (-35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546)
32: (-35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8387909, 45.8387985)
33: (-63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3612823, 55.3612900)
34: (-57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5909424, 47.5909424)
35: (-56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8676910, 44.8676949)
36: (-53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4313049, 49.4312973)
37: (-78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.8089600, 60.8089523)
38: (-63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6764984, 59.6764984)
39: (-72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0168457, 58.0168381)
40: (-51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460)
41: (-40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611)
42: (-26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.24 + 58.30 = 60.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -20.2038465, upper bound: 20.2038465

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1757
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1757

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1549411, upper bound: 20.2023272
time: 51.66 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.2023272, upper bound: 20.1549411
time: 48.94 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 100.72 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 100.72
Output dim: 5, lower bound: -20.1549411, upper bound: 20.2023272
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 100.72
Output dim: 5, lower bound: -20.2023272, upper bound: 20.1549411

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4060974, 38.4061050
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9550018, 41.9550934
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1094666, 36.1097336
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5953522, 38.5960236
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7565231, 50.7567368
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2178040, 45.2174606
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5444336, 59.5437012
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0679092, 56.0673599
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4200821, 38.4196548
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8397980, 45.8395538
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3580475, 55.3576279
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5763550, 47.5743179
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8621063, 44.8614349
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4408112, 49.4402618
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.8029327, 60.8023682
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6934967, 59.6929398
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0216522, 58.0208740
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1024880, upper bound: 20.1989167
time: 57.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1515333, upper bound: 20.1498606
time: 51.64 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4061050, 38.4061012
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9550934, 41.9549980
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1097412, 36.1094666
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5960236, 38.5953522
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7567368, 50.7565231
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2174683, 45.2178116
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5437012, 59.5444336
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0673599, 56.0679169
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4196548, 38.4200821
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8395538, 45.8398056
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3576202, 55.3580475
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5743103, 47.5763550
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8614349, 44.8621063
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4402618, 49.4408112
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.8023682, 60.8029327
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6929474, 59.6935043
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0208740, 58.0216522
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1671
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1671

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1498606, upper bound: 20.1515333
time: 53.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1989167, upper bound: 20.1024880
time: 78.83 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 134.45 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 134.45
Output dim: 5, lower bound: -20.1024880, upper bound: 20.1989167
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 134.45
Output dim: 5, lower bound: -20.1515333, upper bound: 20.1498606
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 134.45
Output dim: 5, lower bound: -20.1498606, upper bound: 20.1515333
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 134.45
Output dim: 5, lower bound: -20.1989167, upper bound: 20.1024880

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4036560, 38.4050636
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9545212, 41.9547462
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1095695, 36.1096878
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5954514, 38.5958633
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7585678, 50.7565460
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2177734, 45.2176208
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5434723, 59.5457687
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0655670, 56.0640526
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4200821, 38.4196548
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8395538, 45.8417130
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3515320, 55.3559036
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5699463, 47.5713577
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8568573, 44.8585854
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4399567, 49.4403534
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.8024902, 60.8019943
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6934967, 59.6929474
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0199738, 58.0238037
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0586482, upper bound: 20.1976153
time: 48.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1011706, upper bound: 20.1544479
time: 46.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4050598, 38.4036560
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9546509, 41.9546165
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1094246, 36.1098366
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5951920, 38.5961227
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7563248, 50.7587814
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2179718, 45.2174225
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5464935, 59.5427475
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0646210, 56.0650253
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4200821, 38.4196548
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8419647, 45.8393097
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3563232, 55.3511047
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5733948, 47.5679245
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8592606, 44.8561859
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4409027, 49.4394150
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.8025513, 60.8019257
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6934967, 59.6929245
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0245819, 58.0191956
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1076748, upper bound: 20.1485593
time: 49.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1502152, upper bound: 20.1053996
time: 48.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4036560, 38.4050598
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9546204, 41.9546509
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1098366, 36.1094208
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5961227, 38.5951920
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7587814, 50.7563248
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2174225, 45.2179718
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5427399, 59.5465012
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0650177, 56.0646095
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4196548, 38.4200821
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8393097, 45.8419647
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3511047, 55.3563309
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5679321, 47.5733948
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8561859, 44.8592606
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4394073, 49.4408951
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.8019257, 60.8025589
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6929169, 59.6935043
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0191956, 58.0245819
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1053996, upper bound: 20.1502152
time: 61.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1485593, upper bound: 20.1076748
time: 89.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4050674, 38.4036560
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9547424, 41.9545212
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1096916, 36.1095657
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5958633, 38.5954552
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7565384, 50.7585602
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2176208, 45.2177734
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5457611, 59.5434799
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0640411, 56.0655823
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4196548, 38.4200821
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8417206, 45.8395615
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3558960, 55.3515320
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5713501, 47.5699539
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8585892, 44.8568573
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4403534, 49.4399567
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.8019867, 60.8024902
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6929474, 59.6934814
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0238037, 58.0199661
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1758
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1758

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1544479, upper bound: 20.1011706
time: 55.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1976153, upper bound: 20.0586482
time: 43.25 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 100.31 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 100.31
Output dim: 5, lower bound: -20.0586482, upper bound: 20.1976153
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 100.31
Output dim: 5, lower bound: -20.1011706, upper bound: 20.1544479
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 100.31
Output dim: 5, lower bound: -20.1076748, upper bound: 20.1485593
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 100.31
Output dim: 5, lower bound: -20.1502152, upper bound: 20.1053996
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 100.31
Output dim: 5, lower bound: -20.1053996, upper bound: 20.1502152
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 100.31
Output dim: 5, lower bound: -20.1485593, upper bound: 20.1076748
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 100.31
Output dim: 5, lower bound: -20.1544479, upper bound: 20.1011706
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 100.31
Output dim: 5, lower bound: -20.1976153, upper bound: 20.0586482

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.3995285, 38.4015312
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9498901, 41.9508247
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1023712, 36.1036224
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5846100, 38.5867271
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7533264, 50.7521591
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2045670, 45.2014084
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5540009, 59.5546417
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0935059, 56.0875206
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4165344, 38.4154434
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8397675, 45.8419113
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3392792, 55.3413010
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5497742, 47.5472794
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8384399, 44.8366547
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4320984, 49.4310532
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7840576, 60.7801590
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6842194, 59.6820297
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0118027, 58.0141830
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0287720, upper bound: 20.1967663
time: 70.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0576644, upper bound: 20.1483650
time: 55.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4001160, 38.4009399
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9505997, 41.9501114
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1035004, 36.1024933
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5863190, 38.5850182
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7541809, 50.7513123
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2015762, 45.2044144
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5523529, 59.5562820
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0890503, 56.0919800
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4158707, 38.4161072
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8397675, 45.8419342
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3369293, 55.3436508
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5458679, 47.5511780
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8349304, 44.8401642
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4306488, 49.4324951
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7806549, 60.7835541
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6825714, 59.6836777
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0103378, 58.0156479
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.10 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0728087, upper bound: 20.1536094
time: 51.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1002480, upper bound: 20.1047647
time: 47.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4009323, 38.4001236
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9500198, 41.9506989
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1022263, 36.1037712
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5843430, 38.5869904
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7510986, 50.7543945
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2047653, 45.2012177
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5570221, 59.5516205
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0925293, 56.0884933
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4165344, 38.4154434
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8421783, 45.8395081
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3440857, 55.3365021
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5532227, 47.5438385
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8408356, 44.8342552
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4330292, 49.4301147
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7841187, 60.7800980
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6842346, 59.6820068
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0164413, 58.0095749
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0777988, upper bound: 20.1477103
time: 52.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1066910, upper bound: 20.0993206
time: 45.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4015274, 38.3995323
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9507294, 41.9499817
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1033554, 36.1026421
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5860596, 38.5852776
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7519379, 50.7535477
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2017593, 45.2042160
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5553741, 59.5532532
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0880737, 56.0929565
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4158707, 38.4161072
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8421631, 45.8395309
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3417206, 55.3388519
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5493164, 47.5477448
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8373260, 44.8377647
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4315948, 49.4315567
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7807159, 60.7834930
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6825867, 59.6836548
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0149765, 58.0110321
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0728087, upper bound: 20.1045625
time: 54.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1002480, upper bound: 20.0557366
time: 54.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.3995361, 38.4015274
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9499817, 41.9507294
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1026459, 36.1033554
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5852737, 38.5860596
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7535400, 50.7519455
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2042160, 45.2017670
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5532684, 59.5553741
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0929565, 56.0880775
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4161072, 38.4158707
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8395233, 45.8421631
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3388519, 55.3417282
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5477448, 47.5493088
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8377686, 44.8373260
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4315643, 49.4315948
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7834930, 60.7807236
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6836548, 59.6825943
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0110397, 58.0149612
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0557366, upper bound: 20.1492937
time: 56.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1045625, upper bound: 20.1218374
time: 46.50 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4001236, 38.4009323
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9506989, 41.9500198
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1037750, 36.1022263
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5869904, 38.5843468
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7543945, 50.7510986
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2012100, 45.2047653
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5516205, 59.5570068
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0885010, 56.0925369
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4154434, 38.4165344
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8395081, 45.8421783
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3365021, 55.3440781
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5438538, 47.5532150
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8342590, 44.8408394
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4301147, 49.4330368
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7800903, 60.7841187
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6820068, 59.6842422
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0095749, 58.0164261
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0993206, upper bound: 20.1066910
time: 51.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1477103, upper bound: 20.0777988
time: 45.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4009399, 38.4001198
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9501114, 41.9505997
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1024933, 36.1035004
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5850143, 38.5863190
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7513123, 50.7541809
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2044144, 45.2015686
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5562897, 59.5523529
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0919800, 56.0890503
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4161072, 38.4158707
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8419342, 45.8397598
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3436584, 55.3369217
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5511780, 47.5458755
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8401642, 44.8349266
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4324799, 49.4306641
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7835541, 60.7806625
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6836853, 59.6825638
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0156479, 58.0103531
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1047647, upper bound: 20.1002480
time: 54.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1536094, upper bound: 20.0728087
time: 57.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4015350, 38.3995285
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9508209, 41.9498901
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.1036224, 36.1023712
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5867233, 38.5846100
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7521515, 50.7533340
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2014084, 45.2045670
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5546417, 59.5539856
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0875244, 56.0935097
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4154434, 38.4165344
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8419037, 45.8397827
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3413086, 55.3392792
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5472717, 47.5497818
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8366547, 44.8384399
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4310455, 49.4320984
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7801514, 60.7840576
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6820374, 59.6842117
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0141830, 58.0118103
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1773
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1773

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1483650, upper bound: 20.0576644
time: 67.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1967663, upper bound: 20.0287720
time: 75.44 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 145.38 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 145.38
Output dim: 5, lower bound: -20.0287720, upper bound: 20.1967663
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 145.38
Output dim: 5, lower bound: -20.0576644, upper bound: 20.1483650
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 145.38
Output dim: 5, lower bound: -20.0728087, upper bound: 20.1536094
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 145.38
Output dim: 5, lower bound: -20.1002480, upper bound: 20.1047647
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 145.38
Output dim: 5, lower bound: -20.0777988, upper bound: 20.1477103
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 145.38
Output dim: 5, lower bound: -20.1066910, upper bound: 20.0993206
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 145.38
Output dim: 5, lower bound: -20.0728087, upper bound: 20.1045625
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 145.38
Output dim: 5, lower bound: -20.1002480, upper bound: 20.0557366
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 145.38
Output dim: 5, lower bound: -20.0557366, upper bound: 20.1492937
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 145.38
Output dim: 5, lower bound: -20.1045625, upper bound: 20.1218374
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 145.38
Output dim: 5, lower bound: -20.0993206, upper bound: 20.1066910
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 145.38
Output dim: 5, lower bound: -20.1477103, upper bound: 20.0777988
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 145.38
Output dim: 5, lower bound: -20.1047647, upper bound: 20.1002480
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 145.38
Output dim: 5, lower bound: -20.1536094, upper bound: 20.0728087
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 145.38
Output dim: 5, lower bound: -20.1483650, upper bound: 20.0576644
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 145.38
Output dim: 5, lower bound: -20.1967663, upper bound: 20.0287720

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4004974, 38.4024429
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9479446, 41.9491997
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0977249, 36.0997543
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5773544, 38.5806923
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7478027, 50.7475281
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2054901, 45.2004547
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5650024, 59.5642548
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.1067963, 56.0988731
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4144592, 38.4129486
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8362579, 45.8379288
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3253860, 55.3247986
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5304260, 47.5240784
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8224945, 44.8175888
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4264221, 49.4245682
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7736664, 60.7678452
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6791382, 59.6766510
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0015411, 58.0023270
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0273819, upper bound: 20.1551387
time: 44.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -19.9871358, upper bound: 20.1953748
time: 56.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4004364, 38.4025002
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9482651, 41.9488831
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0985031, 36.0989761
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5785751, 38.5794716
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7486725, 50.7466278
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2036133, 45.2022858
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5635986, 59.5653076
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.1048584, 56.1008034
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4140396, 38.4133720
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8358002, 45.8383789
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3227615, 55.3274002
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5265808, 47.5279312
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8193741, 44.8207092
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4256287, 49.4251938
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7717438, 60.7697067
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6788483, 59.6768646
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9999542, 58.0036469
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0562743, upper bound: 20.1067373
time: 55.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -19.9871358, upper bound: 20.1469745
time: 62.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4010849, 38.4018478
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9486618, 41.9484863
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0988541, 36.0986252
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5790634, 38.5789833
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7486572, 50.7466888
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2024841, 45.2034454
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5633850, 59.5658951
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.1023407, 56.1033325
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4137955, 38.4136124
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8362427, 45.8379517
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3230362, 55.3271484
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5265350, 47.5279846
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8189774, 44.8210983
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4249878, 49.4260101
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7702789, 60.7712402
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6774902, 59.6782990
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0000763, 58.0037842
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0714184, upper bound: 20.1119821
time: 71.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0311732, upper bound: 20.1522187
time: 54.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4010315, 38.4019089
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9489746, 41.9481735
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0996323, 36.0978470
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5802841, 38.5777626
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7495117, 50.7457886
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2006073, 45.2052917
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5619659, 59.5669479
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.1004028, 56.1052628
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4133759, 38.4140358
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8357849, 45.8384018
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3204117, 55.3297501
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5226746, 47.5318375
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8158646, 44.8242188
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4241791, 49.4266357
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7683411, 60.7730942
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6772003, 59.6785126
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9984894, 58.0051041
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0988572, upper bound: 20.0631284
time: 55.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0160281, upper bound: 20.1033746
time: 46.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4019012, 38.4010315
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9480743, 41.9490738
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0975723, 36.0999031
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5770874, 38.5809555
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7455750, 50.7497711
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2056885, 45.2002563
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5680237, 59.5612335
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.1058197, 56.0998459
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4144669, 38.4129486
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8386688, 45.8355331
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3301773, 55.3199921
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5338745, 47.5206375
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8248978, 44.8151894
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4273682, 49.4236374
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7737274, 60.7677841
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6791534, 59.6766281
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0061798, 57.9977036
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0273819, upper bound: 20.1060826
time: 47.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0361624, upper bound: 20.1463188
time: 53.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4018478, 38.4010925
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9483948, 41.9487534
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0983582, 36.0991211
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5783081, 38.5797348
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7464294, 50.7488708
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2037964, 45.2020950
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5666199, 59.5622864
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.1038971, 56.1017761
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4140396, 38.4133720
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8382111, 45.8359756
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3275833, 55.3226013
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5300293, 47.5244980
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8217697, 44.8183098
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4265442, 49.4242630
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7718048, 60.7696381
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6788635, 59.6768417
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0045624, 57.9990311
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1053009, upper bound: 20.0576939
time: 57.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0650547, upper bound: 20.0979299
time: 49.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4024963, 38.4004440
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9487839, 41.9483604
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0987015, 36.0987740
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5788040, 38.5792427
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7464142, 50.7489243
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2026825, 45.2032547
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5664062, 59.5628662
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.1013641, 56.1043091
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4137955, 38.4136124
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8386383, 45.8355484
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3278275, 55.3223419
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5299683, 47.5245438
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8213806, 44.8186989
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4259186, 49.4250717
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7703400, 60.7711792
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6775055, 59.6782761
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0047150, 57.9991684
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0764088, upper bound: 20.0629358
time: 50.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0311732, upper bound: 20.1031717
time: 56.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4024353, 38.4005013
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9491043, 41.9480438
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0994873, 36.0979919
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5800247, 38.5780220
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7472839, 50.7480240
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2008057, 45.2050934
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5649872, 59.5639191
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0994263, 56.1062355
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4133759, 38.4140320
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8381805, 45.8359985
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3252335, 55.3249512
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5261230, 47.5284042
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8182602, 44.8218193
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4251099, 49.4256973
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7684021, 60.7730331
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6772156, 59.6784897
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0030975, 58.0004959
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1479032, upper bound: 20.0141018
time: 58.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1076638, upper bound: 20.0543463
time: 51.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4005051, 38.4024353
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9480438, 41.9491043
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0979919, 36.0994873
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5780182, 38.5800247
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7480164, 50.7472839
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2050934, 45.2008057
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5639191, 59.5649872
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.1062317, 56.0994301
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4140320, 38.4133759
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8359985, 45.8381805
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3249588, 55.3252182
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5284119, 47.5261154
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8218155, 44.8182602
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4256897, 49.4251175
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7730408, 60.7684097
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6784973, 59.6772156
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0005035, 58.0031052
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0543463, upper bound: 20.1076638
time: 57.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0141018, upper bound: 20.1479032
time: 53.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4004440, 38.4024963
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9483566, 41.9487877
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0987778, 36.0987053
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5792389, 38.5788040
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7489166, 50.7464142
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2032471, 45.2026825
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5628662, 59.5663986
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.1043091, 56.1013565
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4136124, 38.4137993
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8355560, 45.8386459
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3223343, 55.3278198
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5245361, 47.5299683
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8187027, 44.8213806
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4250793, 49.4259186
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7711792, 60.7703400
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6782837, 59.6775055
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9991608, 58.0047073
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1031717, upper bound: 20.0801999
time: 81.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0629358, upper bound: 20.1204473
time: 64.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4010925, 38.4018440
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9487534, 41.9483948
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0991211, 36.0983582
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5797348, 38.5783119
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7488708, 50.7464371
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2020874, 45.2038040
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5622864, 59.5666199
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.1017761, 56.1038895
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4133682, 38.4140396
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8359833, 45.8382034
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3226089, 55.3275757
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5245056, 47.5300217
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8183060, 44.8217735
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4242554, 49.4265518
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7696381, 60.7718048
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6768494, 59.6788635
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9990387, 58.0045624
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0576939, upper bound: 20.0650547
time: 53.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0576939, upper bound: 20.1053009
time: 50.10 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4010315, 38.4019012
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9490738, 41.9480743
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0999069, 36.0975761
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5809555, 38.5770912
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7497711, 50.7455673
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2002563, 45.2056885
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5612335, 59.5680313
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0998383, 56.1058159
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4129486, 38.4144630
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8355255, 45.8386612
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3199844, 55.3301773
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5206299, 47.5338745
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8151932, 44.8248940
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4236450, 49.4273605
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7677765, 60.7737274
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6766357, 59.6791611
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9976959, 58.0061722
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1060826, upper bound: 20.0361624
time: 51.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1060826, upper bound: 20.0764088
time: 52.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4019089, 38.4010315
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9481735, 41.9489746
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0978470, 36.0996323
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5777588, 38.5802841
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7457886, 50.7495193
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2052917, 45.2006073
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5669556, 59.5619659
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.1052704, 56.1004028
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4140396, 38.4133759
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8383942, 45.8357773
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3297501, 55.3204193
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5318298, 47.5226822
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8242188, 44.8158607
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4266357, 49.4241791
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7731018, 60.7683487
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6785126, 59.6771851
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0051117, 57.9984818
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1033746, upper bound: 20.0586188
time: 44.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.0631284, upper bound: 20.0988572
time: 47.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4018478, 38.4010887
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9484863, 41.9486618
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0986252, 36.0988541
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5789795, 38.5790634
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7466888, 50.7486572
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2034454, 45.2024918
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5658875, 59.5633774
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.1033325, 56.1023331
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4136124, 38.4137993
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8379517, 45.8362427
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3271561, 55.3230209
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5279846, 47.5265350
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8210983, 44.8189812
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4260101, 49.4249802
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7712402, 60.7702789
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6782990, 59.6774902
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0037689, 58.0000916
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1522187, upper bound: 20.0311732
time: 49.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1119821, upper bound: 20.0714184
time: 45.37 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4025040, 38.4004364
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9488831, 41.9482651
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0989761, 36.0985031
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5794678, 38.5785751
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7466278, 50.7486725
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2022858, 45.2036057
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5653076, 59.5635986
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.1007996, 56.1048622
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4133682, 38.4140396
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8383789, 45.8358002
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3274002, 55.3227692
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5279236, 47.5265808
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8207092, 44.8193741
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4252014, 49.4256210
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7696991, 60.7717438
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6768646, 59.6788330
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0036469, 57.9999466
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1469745, upper bound: 20.0160281
time: 57.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1067373, upper bound: 20.0562743
time: 77.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.4024429, 38.4004974
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9491959, 41.9479446
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0997543, 36.0977249
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5806885, 38.5773544
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7475281, 50.7478027
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.2004547, 45.2054901
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5642548, 59.5650101
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0988770, 56.1067924
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4129486, 38.4144592
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8379364, 45.8362656
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3248062, 55.3253784
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5240784, 47.5304413
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.8175888, 44.8224945
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4245605, 49.4264145
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7678528, 60.7736664
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6766510, 59.6791382
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -58.0023346, 58.0015488
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1662
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1662

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1953748, upper bound: 19.9871358
time: 59.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -20.1551387, upper bound: 20.0273819
time: 58.33 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 119.70 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0273819, upper bound: 20.1551387
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -19.9871358, upper bound: 20.1953748
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0562743, upper bound: 20.1067373
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -19.9871358, upper bound: 20.1469745
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0714184, upper bound: 20.1119821
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0311732, upper bound: 20.1522187
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0988572, upper bound: 20.0631284
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0160281, upper bound: 20.1033746
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0273819, upper bound: 20.1060826
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0361624, upper bound: 20.1463188
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.1053009, upper bound: 20.0576939
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0650547, upper bound: 20.0979299
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0764088, upper bound: 20.0629358
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0311732, upper bound: 20.1031717
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.1479032, upper bound: 20.0141018
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.1076638, upper bound: 20.0543463
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0543463, upper bound: 20.1076638
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0141018, upper bound: 20.1479032
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.1031717, upper bound: 20.0801999
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0629358, upper bound: 20.1204473
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0576939, upper bound: 20.0650547
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0576939, upper bound: 20.1053009
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.1060826, upper bound: 20.0361624
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.1060826, upper bound: 20.0764088
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.1033746, upper bound: 20.0586188
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.0631284, upper bound: 20.0988572
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.1522187, upper bound: 20.0311732
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.1119821, upper bound: 20.0714184
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.1469745, upper bound: 20.0160281
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.1067373, upper bound: 20.0562743
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.1953748, upper bound: 19.9871358
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 119.70
Output dim: 5, lower bound: -20.1551387, upper bound: 20.0273819

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -37.7161484, 17.6098995, -37.7161484, 17.6098995, -55.3260498, 55.3260498
1: -11.9740734, 22.4840317, -11.9740734, 22.4840317, -34.4581070, 34.4581070
2: -9.7768641, 25.2916794, -9.7768641, 25.2916794, -35.0685425, 35.0685425
3: -9.6581364, 28.9738884, -9.6581364, 28.9738884, -38.3929825, 38.3934784
4: -16.6948967, 25.3766670, -16.6948967, 25.3766670, -41.9411926, 41.9411659
5: -7.4954538, 29.0416183, -7.4954538, 29.0416183, -36.0882416, 36.0884857
6: -38.2646103, 12.0445700, -38.2646103, 12.0445700, -50.3091812, 50.3091812
7: -11.1436720, 28.6731205, -11.1436720, 28.6731205, -38.5648956, 38.5658875
8: -21.3061943, 29.8744297, -21.3061943, 29.8744297, -50.7381668, 50.7361450
9: -13.7687559, 28.3488617, -13.7687559, 28.3488617, -42.1176186, 42.1176186
10: -22.1398239, 32.0396423, -22.1398239, 32.0396423, -54.1794662, 54.1794662
11: -23.7499428, 14.7507343, -23.7499428, 14.7507343, -38.5006790, 38.5006790
12: -44.2741814, 4.4801540, -44.2741814, 4.4801540, -45.1728821, 45.1730804
13: -37.4942932, 22.3672791, -37.4942932, 22.3672791, -59.5876770, 59.5914459
14: -64.9316254, 2.7553263, -64.9316254, 2.7553263, -67.6869507, 67.6869507
15: -21.9280663, 20.3756504, -21.9280663, 20.3756504, -42.3037186, 42.3037186
16: -23.4872456, 21.7699547, -23.4872456, 21.7699547, -45.2572021, 45.2572021
17: -58.4270172, -1.1384592, -58.4270172, -1.1384592, -56.0539017, 56.0541077
18: -35.8796768, 14.6562901, -35.8796768, 14.6562901, -50.5359650, 50.5359650
19: -26.4627781, 9.5100994, -26.4627781, 9.5100994, -35.9728775, 35.9728775
20: -21.5785027, 15.9173069, -21.5785027, 15.9173069, -37.4958115, 37.4958115
21: -27.3156834, 13.0029221, -27.3156834, 13.0029221, -40.3186035, 40.3186035
22: -32.1411972, 10.6446962, -32.1411972, 10.6446962, -42.7858925, 42.7858925
23: -24.6154861, 14.0575972, -24.6154861, 14.0575972, -38.6730843, 38.6730843
24: -30.7798500, 13.7447214, -30.7798500, 13.7447214, -44.5245705, 44.5245705
25: -28.9277706, 12.9479380, -28.9277706, 12.9479380, -41.8757095, 41.8757095
26: -41.0683975, 17.0873718, -41.0683975, 17.0873718, -58.1557693, 58.1557693
27: -26.1438465, 18.1995564, -26.1438465, 18.1995564, -44.3434029, 44.3434029
28: -25.1042480, 17.3391590, -25.1042480, 17.3391590, -42.4434052, 42.4434052
29: -27.6379871, 10.9656658, -27.6379871, 10.9656658, -38.4094391, 38.4087296
30: -26.8832150, 18.3511600, -26.8832150, 18.3511600, -45.2343750, 45.2343750
31: -35.4411011, 12.1511536, -35.4411011, 12.1511536, -47.5922546, 47.5922546
32: -35.2567940, 11.0384121, -35.2567940, 11.0384121, -45.8358459, 45.8375320
33: -63.7363930, -3.7368479, -63.7363930, -3.7368479, -55.3113632, 55.3141479
34: -57.8639908, -6.3502693, -57.8639908, -6.3502693, -47.5004730, 47.4987717
35: -56.1062355, -4.3361292, -56.1062355, -4.3361292, -44.7963409, 44.7968979
36: -53.5096817, 0.8973608, -53.5096817, 0.8973608, -49.4048462, 49.4072952
37: -78.3080139, -14.2384996, -78.3080139, -14.2384996, -60.7380524, 60.7380371
38: -63.8501053, 0.4308362, -63.8501053, 0.4308362, -59.6470184, 59.6506195
39: -72.1730728, -8.1493416, -72.1730728, -8.1493416, -57.9862061, 57.9906693
40: -51.3974609, -6.1964159, -51.3974609, -6.1964159, -45.2010460, 45.2010460
41: -40.0853310, 12.2757301, -40.0853310, 12.2757301, -52.3610611, 52.3610611
42: -26.1899185, 11.9887085, -26.1899185, 11.9887085, -38.1786270, 38.1786270

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1655
type: RSZ, layer: 1, pos: 733
type: RSZ, layer: 1, pos: 1725
type: RSZ, layer: 1, pos: 1774
type: RSZ, layer: 1, pos: 734
type: RSZ, layer: 1, pos: 1787
type: RSZ, layer: 1, pos: 1726
type: RSZ, layer: 1, pos: 1741
type: RSZ, layer: 1, pos: 735
type: RSZ, layer: 1, pos: 1771
type: RSZ, layer: 1, pos: 723
type: RSZ, layer: 1, pos: 1742
type: RSZ, layer: 1, pos: 1788
type: RSZ, layer: 1, pos: 1759
type: RSZ, layer: 1, pos: 1772
type: RSZ, layer: 1, pos: 535
type: RSZ, layer: 1, pos: 533
type: RSZ, layer: 1, pos: 765
type: RSZ, layer: 1, pos: 1779
type: RSZ, layer: 1, pos: 1663
type: RSZ, layer: 1, pos: 1650
type: RSZ, layer: 1, pos: 1722
type: RSZ, layer: 1, pos: 1401
type: RSZ, layer: 1, pos: 755
type: RSZ, layer: 1, pos: 1718
type: RSZ, layer: 1, pos: 749
type: RSZ, layer: 1, pos: 1646
type: RSZ, layer: 1, pos: 1755
type: RSZ, layer: 1, pos: 1678
type: RSZ, layer: 1, pos: 717
type: RSZ, layer: 1, pos: 739
type: RSZ, layer: 1, pos: 534
type: RSZ, layer: 1, pos: 893
type: RSZ, layer: 1, pos: 722
type: RSZ, layer: 1, pos: 721
type: RSZ, layer: 1, pos: 718
type: RSZ, layer: 1, pos: 1555
type: RSZ, layer: 1, pos: 518
type: RSZ, layer: 1, pos: 1727
type: RSZ, layer: 1, pos: 753
type: RSZ, layer: 1, pos: 1630
type: RSZ, layer: 1, pos: 732
type: RSZ, layer: 1, pos: 646
type: RSZ, layer: 1, pos: 1416
type: RSZ, layer: 1, pos: 750
type: RSZ, layer: 1, pos: 702
type: RSZ, layer: 1, pos: 767
type: RSZ, layer: 1, pos: 766
type: RSZ, layer: 1, pos: 737
type: RSZ, layer: 1, pos: 703
type: RSZ, layer: 1, pos: 751
type: RSZ, layer: 1, pos: 671
type: RSZ, layer: 1, pos: 630
type: RSZ, layer: 1, pos: 909
type: RSZ, layer: 1, pos: 1775
type: RSZ, layer: 1, pos: 1400
type: RSZ, layer: 1, pos: 738
type: RSZ, layer: 1, pos: 1743
type: RSZ, layer: 1, pos: 532
type: RSZ, layer: 1, pos: 1417
type: RSZ, layer: 1, pos: 1687
type: RSZ, layer: 1, pos: 1402
type: RSZ, layer: 1, pos: 754
type: RSZ, layer: 1, pos: 764
type: RSZ, layer: 1, pos: 1679
type: RSZ, layer: 1, pos: 1556
type: RSZ, layer: 1, pos: 655
type: RSZ, layer: 1, pos: 748
type: RSZ, layer: 1, pos: 706
type: RSZ, layer: 1, pos: 759
type: RSZ, layer: 1, pos: 1707
type: RSZ, layer: 1, pos: 761
type: RSZ, layer: 1, pos: 1647
type: RSZ, layer: 1, pos: 1557
type: RSZ, layer: 1, pos: 1680
type: RSZ, layer: 1, pos: 1589
type: RSZ, layer: 1, pos: 1539
type: RSZ, layer: 1, pos: 740
type: RSZ, layer: 1, pos: 1694
type: RSZ, layer: 1, pos: 1418
type: RSZ, layer: 1, pos: 741
type: RSZ, layer: 1, pos: 877
type: RSZ, layer: 1, pos: 1541
type: RSZ, layer: 1, pos: 1634
type: RSZ, layer: 1, pos: 756
type: RSZ, layer: 1, pos: 531
type: RSZ, layer: 1, pos: 1696
type: RSZ, layer: 1, pos: 705
type: RSZ, layer: 1, pos: 925
type: RSZ, layer: 1, pos: 681
type: RSZ, layer: 1, pos: 1608
type: RSZ, layer: 1, pos: 1429
type: RSZ, layer: 1, pos: 935
type: RSZ, layer: 1, pos: 561
type: RSZ, layer: 1, pos: 747
type: RSZ, layer: 1, pos: 1689
type: RSZ, layer: 1, pos: 1638
type: RSZ, layer: 1, pos: 757
type: RSZ, layer: 1, pos: 673
type: RSZ, layer: 1, pos: 898
type: RSZ, layer: 1, pos: 1540
type: RSZ, layer: 1, pos: 760
type: RSZ, layer: 1, pos: 1554
type: RSZ, layer: 1, pos: 1558
type: RSZ, layer: 1, pos: 1560
type: RSZ, layer: 1, pos: 1639
type: RSZ, layer: 1, pos: 1573
type: RSZ, layer: 1, pos: 577
type: RSZ, layer: 1, pos: 578
type: RSZ, layer: 1, pos: 1637
type: RSZ, layer: 1, pos: 745
type: RSZ, layer: 1, pos: 1732
type: RSZ, layer: 1, pos: 763
type: RSZ, layer: 1, pos: 1712
type: RSZ, layer: 1, pos: 672
type: RSZ, layer: 1, pos: 688
type: RSZ, layer: 1, pos: 530
type: RSZ, layer: 1, pos: 1542
type: RSZ, layer: 1, pos: 1586
type: RSZ, layer: 1, pos: 1446
type: RSZ, layer: 1, pos: 560
type: RSZ, layer: 1, pos: 933
type: RSZ, layer: 1, pos: 736
type: RSZ, layer: 1, pos: 982
type: RSZ, layer: 1, pos: 656
type: RSZ, layer: 1, pos: 1451
type: RSZ, layer: 1, pos: 752
type: RSZ, layer: 1, pos: 1431
type: RSZ, layer: 1, pos: 956
type: RSZ, layer: 1, pos: 1618
type: RSZ, layer: 1, pos: 883
type: RSZ, layer: 1, pos: 1575
type: RSZ, layer: 1, pos: 544
type: RSZ, layer: 1, pos: 1760
type: RSZ, layer: 1, pos: 914
type: RSZ, layer: 1, pos: 1587
type: RSZ, layer: 1, pos: 882
type: RSZ, layer: 1, pos: 1340
type: RSZ, layer: 1, pos: 859
type: RSZ, layer: 1, pos: 938
type: RSZ, layer: 1, pos: 779
type: RSZ, layer: 1, pos: 1561
type: RSZ, layer: 1, pos: 545
type: RSZ, layer: 1, pos: 899
type: RSZ, layer: 1, pos: 1619
type: RSZ, layer: 1, pos: 1783
type: RSZ, layer: 1, pos: 520
type: RSZ, layer: 1, pos: 860
type: RSZ, layer: 1, pos: 1568
type: RSZ, layer: 1, pos: 758
type: RSZ, layer: 1, pos: 1744
type: RSZ, layer: 1, pos: 1776
type: RSZ, layer: 1, pos: 1538
type: RSZ, layer: 1, pos: 1543
type: RSZ, layer: 1, pos: 514
type: RSZ, layer: 1, pos: 1433
type: RSZ, layer: 1, pos: 1784
type: RSZ, layer: 1, pos: 762
type: RSZ, layer: 1, pos: 866
type: RSZ, layer: 1, pos: 1387
type: RSZ, layer: 1, pos: 1617
type: RSZ, layer: 1, pos: 843
type: RSZ, layer: 1, pos: 563
type: RSZ, layer: 1, pos: 1374
type: RSZ, layer: 1, pos: 823
type: RSZ, layer: 1, pos: 971
type: RSZ, layer: 1, pos: 780
type: RSZ, layer: 1, pos: 1664
type: RSZ, layer: 1, pos: 1537
type: RSZ, layer: 1, pos: 592
type: RSZ, layer: 1, pos: 1695
type: RSZ, layer: 1, pos: 1559
type: RSZ, layer: 1, pos: 1544
type: RSZ, layer: 1, pos: 1571
type: RSZ, layer: 1, pos: 851
type: RSZ, layer: 1, pos: 529
type: RSZ, layer: 1, pos: 548
type: RSZ, layer: 1, pos: 949
type: RSZ, layer: 1, pos: 1584
type: RSZ, layer: 1, pos: 746
type: RSZ, layer: 1, pos: 1553
type: RSZ, layer: 1, pos: 1642
type: RSZ, layer: 1, pos: 1570
type: RSZ, layer: 1, pos: 1448
type: RSZ, layer: 1, pos: 1434
type: RSZ, layer: 1, pos: 824
type: RSZ, layer: 1, pos: 947
type: RSZ, layer: 1, pos: 850
type: RSZ, layer: 1, pos: 1578
type: RSZ, layer: 1, pos: 777
type: RSZ, layer: 1, pos: 720
type: RSZ, layer: 1, pos: 967
type: RSZ, layer: 1, pos: 948
type: RSZ, layer: 1, pos: 546
type: RSZ, layer: 1, pos: 1357
type: RSZ, layer: 1, pos: 528
type: RSZ, layer: 1, pos: 1414
type: RSZ, layer: 1, pos: 931
type: RSZ, layer: 1, pos: 1601
type: RSZ, layer: 1, pos: 1602
type: RSZ, layer: 1, pos: 835
type: RSZ, layer: 1, pos: 778
type: RSZ, layer: 1, pos: 867
type: RSZ, layer: 1, pos: 822
type: RSZ, layer: 1, pos: 624
type: RSZ, layer: 1, pos: 1616
type: RSZ, layer: 1, pos: 562
type: RSZ, layer: 1, pos: 576
type: RSZ, layer: 1, pos: 1588
type: RSZ, layer: 1, pos: 1574
type: RSZ, layer: 1, pos: 809
type: RSZ, layer: 1, pos: 1536
type: RSZ, layer: 1, pos: 1585
type: RSZ, layer: 1, pos: 915
type: RSZ, layer: 1, pos: 820
type: RSZ, layer: 1, pos: 808
type: RSZ, layer: 1, pos: 1648
type: RSZ, layer: 1, pos: 1576
type: RSZ, layer: 1, pos: 1711
type: RSZ, layer: 1, pos: 1552
type: RSZ, layer: 1, pos: 547
type: RSZ, layer: 1, pos: 1308
type: RSZ, layer: 1, pos: 1572
type: RSZ, layer: 1, pos: 1710
type: RSZ, layer: 1, pos: 807
type: RSZ, layer: 1, pos: 686
type: RSZ, layer: 1, pos: 876
type: RSZ, layer: 1, pos: 1293
type: RSZ, layer: 1, pos: 964
type: RSZ, layer: 1, pos: 1420
type: RSZ, layer: 1, pos: 1324
type: RSZ, layer: 1, pos: 704
type: RSZ, layer: 1, pos: 825
type: RSZ, layer: 1, pos: 1569
type: RSZ, layer: 1, pos: 1382
type: RSZ, layer: 1, pos: 1728
type: RSZ, layer: 1, pos: 837
type: RSZ, layer: 1, pos: 689
type: RSZ, layer: 1, pos: 1413
type: RSZ, layer: 1, pos: 1358
type: RSZ, layer: 1, pos: 806
type: RSZ, layer: 1, pos: 1786
type: RSZ, layer: 1, pos: 1632
type: RSZ, layer: 1, pos: 1603
type: RSZ, layer: 1, pos: 640
type: RSZ, layer: 1, pos: 852
type: RSZ, layer: 1, pos: 1577
type: RSZ, layer: 1, pos: 941
type: RSZ, layer: 1, pos: 1359
type: RSZ, layer: 1, pos: 1436
type: RSZ, layer: 1, pos: 836
type: RSZ, layer: 1, pos: 1600
type: RSZ, layer: 1, pos: 1404
type: RSZ, layer: 1, pos: 940
type: RSZ, layer: 1, pos: 1384
type: RSZ, layer: 1, pos: 821
type: RSZ, layer: 1, pos: 549
type: RSZ, layer: 1, pos: 970

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 1655

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -19.9743875, upper bound: 20.0853573
time: 34.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -19.9743875, upper bound: 20.0835701
time: 33.65 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 69.84 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 69.84
Output dim: 5, lower bound: -19.9743875, upper bound: 20.0853573
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 69.84
Output dim: 5, lower bound: -19.9743875, upper bound: 20.0835701
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -19.9871358, upper bound: 20.1953748
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.0562743, upper bound: 20.1067373
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -19.9871358, upper bound: 20.1469745
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.0714184, upper bound: 20.1119821
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.0311732, upper bound: 20.1522187
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.0988572, upper bound: 20.0631284
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.0160281, upper bound: 20.1033746
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.0273819, upper bound: 20.1060826
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.0361624, upper bound: 20.1463188
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.1053009, upper bound: 20.0576939
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.0650547, upper bound: 20.0979299
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.0764088, upper bound: 20.0629358
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.0311732, upper bound: 20.1031717
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.1479032, upper bound: 20.0141018
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.1076638, upper bound: 20.0543463
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.0543463, upper bound: 20.1076638
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.0141018, upper bound: 20.1479032
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.1031717, upper bound: 20.0801999
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.0629358, upper bound: 20.1204473
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.0576939, upper bound: 20.0650547
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.0576939, upper bound: 20.1053009
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.1060826, upper bound: 20.0361624
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.1060826, upper bound: 20.0764088
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.1033746, upper bound: 20.0586188
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.0631284, upper bound: 20.0988572
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.1522187, upper bound: 20.0311732
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.1119821, upper bound: 20.0714184
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.1469745, upper bound: 20.0160281
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.1067373, upper bound: 20.0562743
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.1953748, upper bound: 19.9871358
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 69.84
Output dim: 5, lower bound: -20.1551387, upper bound: 20.0273819

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 60.54 + 3580.30 = 3640.84 seconds
