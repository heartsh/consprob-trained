use FeatureCountSets;
impl FeatureCountSets {
pub fn load_trained_score_params() -> FeatureCountSets {
FeatureCountSets {
hairpin_loop_length_counts: [-0.2364569, -0.20297103, -0.15998627, -0.16806048, -0.22693804, -0.24266657, -0.1252747, -0.13358448, -0.17678839, -0.16363539, -0.104935154, -0.12823299, -0.15125789, -0.12982893, -0.063769326, -0.1277831, -0.09239055, -0.08393931, -0.14025271, -0.088906914, -0.045413703, -0.09561436, -0.05711222, -0.027384711, -0.057706904, -0.038121812, -0.012450244, -0.0372948],
bulge_loop_length_counts: [-0.3399576, -0.25290197, -0.22323877, -0.13037924, -0.09354363, -0.098593354, -0.110506564, -0.04763957, -0.12254098, -0.09477655, -0.024039965, -0.06315927, -0.032793842, -0.08191408, 0.006831557, -0.024357516, -0.04367115, -0.05190182, 0.019683696, -0.03154069, -0.0072425907, -0.01598128, 0.031000819, -0.07023729, -0.027196096, 0.011884168, -0.015058791, 0.031160831, -0.005264557, -0.035172638],
interior_loop_length_counts: [-0.17215727, -0.3174686, -0.27719748, -0.30114257, -0.23804238, -0.17704804, -0.23312722, -0.20180076, -0.15183696, -0.18559995, -0.16964227, -0.1742314, -0.12391513, -0.13426149, -0.1445687, -0.20152469, -0.1208922, -0.10166279, -0.08667367, -0.080678605, -0.04627506, -0.0995606, -0.09767218, -0.06102264, -0.07627763, -0.022980683, -0.06433045, -0.053423863, -0.02936399],
interior_loop_length_counts_symm: [0.015577137, -0.083798364, -0.09027128, -0.064172745, -0.07105447, -0.012200093, -0.07225784, -0.031456135, -0.029005192, -0.056070324, -0.05451054, -0.061666336, -0.025390787, 0.058901157, -0.0027828824],
interior_loop_length_counts_asymm: [-0.26209947, -0.061246447, -0.014265338, 0.01923394, -0.03978473, -0.020382443, 0.03393309, 0.041388012, -0.014256492, 0.026362253, -0.050781306, -0.036575343, -0.011301465, 0.016998408, 0.00083032466, -0.0076254173, -0.014026961, -0.06558275, -0.0090829255, -0.06837167, -0.038248643, -0.005107676, -0.01403331, -0.0018873727, 0.015118337, -0.011536599, 0.0036646805, 0.018952059],
stack_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.28011268], [0.0, 0.0, 0.14546816, 0.0], [0.0, 0.19249827, 0.0, 0.07031142], [0.10617974, 0.0, 0.075386375, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.18886366], [0.0, 0.0, 0.26100752, 0.0], [0.0, 0.13190162, 0.0, 0.043830432], [0.19249827, 0.0, 0.074140705, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.13531376], [0.0, 0.0, 0.17435367, 0.0], [0.0, 0.26100752, 0.0, 0.09348197], [0.14546816, 0.0, 0.09479347, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.09561349], [0.0, 0.0, 0.09479347, 0.0], [0.0, 0.074140705, 0.0, 0.0242038], [0.075386375, 0.0, 0.012600698, 0.0]]], [[[0.0, 0.0, 0.0, 0.17827727], [0.0, 0.0, 0.13531376, 0.0], [0.0, 0.18886366, 0.0, 0.01609163], [0.28011268, 0.0, 0.09561349, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.01609163], [0.0, 0.0, 0.09348197, 0.0], [0.0, 0.043830432, 0.0, 0.025692143], [0.07031142, 0.0, 0.0242038, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
terminal_mismatch_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.01603105, 0.03730638, 0.07477393, -0.07799379], [-0.011794941, 0.012812435, -0.06935825, 0.043531757], [0.01846667, -0.0780336, -0.035282735, -0.019168817], [-0.10101148, 0.018222168, -0.066294305, 0.016823024]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.02240539, 0.024402807, 0.033294715, -0.09857288], [-0.0016547915, 0.053426903, -0.03599101, -0.025972579], [0.037384234, 0.012558281, 0.04247659, -0.06119435], [-0.118127316, 0.066788755, -0.031467985, -0.05965616]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.025542755, 0.006961428, -0.021066008, -0.09619528], [0.05535868, 0.008386102, -0.065955825, 0.0019966443], [-0.0073052477, -0.07994249, 0.065709054, 0.052776657], [-0.053309575, 0.0010850689, -0.018058192, 0.022681747]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.06776534, -0.02407716, 0.043972325, -0.05215393], [0.018157309, 0.062763885, -0.039766777, -0.014585436], [0.00077852514, -0.06845544, -0.03580404, 0.025473267], [-0.058163658, 0.03535508, -0.013891391, -0.0060793785]]], [[[0.021288201, 0.027494876, -0.0098839225, -0.054690853], [0.051390015, 0.012384279, -0.07883882, 0.0059418953], [-0.003280415, -0.10481602, -0.060871687, -0.020559605], [-0.078705326, -0.012556626, -0.019452946, 0.015919108]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.017700005, 0.045169488, -0.06876366, -0.020567682], [0.018893015, 0.027413402, -0.0675688, -0.072195776], [0.022376023, -0.021035347, 0.008019586, -0.04084889], [-0.086624146, 0.03947039, -0.00005061383, -0.010319232]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
left_dangle_count_mat: [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.040395435, 0.0080208685, 0.005791715, -0.06798969]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.07244702, 0.01753006, -0.026824627, -0.07604059], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [-0.052033313, -0.10557412, -0.025551513, -0.060716935], [0.0, 0.0, 0.0, 0.0], [-0.019918146, -0.00041989906, -0.03094033, -0.008363006]], [[-0.04819494, -0.038100567, 0.0006285195, -0.08321991], [0.0, 0.0, 0.0, 0.0], [-0.0006839795, 0.0028315596, -0.0067109047, -0.0005803483], [0.0, 0.0, 0.0, 0.0]]],
right_dangle_count_mat: [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.0430898, -0.05398293, -0.0022518907, 0.006495047]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.042093746, 0.11807632, 0.074049965, -0.035433326], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [-0.039184246, -0.009066161, 0.03234922, -0.0067844815], [0.0, 0.0, 0.0, 0.0], [-0.025644425, -0.07015058, 0.010191162, 0.012810324]], [[-0.033439446, -0.034291707, 0.01493408, -0.00454945], [0.0, 0.0, 0.0, 0.0], [-0.028415238, 0.038505603, 0.047888573, 0.025932478], [0.0, 0.0, 0.0, 0.0]]],
helix_end_count_mat: [[0.0, 0.0, 0.0, -0.5816755], [0.0, 0.0, -0.5607569, 0.0], [0.0, -0.63383824, 0.0, -0.20253748], [-0.59629387, 0.0, -0.19580403, 0.0]],
base_pair_count_mat: [[0.0, 0.0, 0.0, 0.7179288], [0.0, 0.0, 0.835359, 0.0], [0.0, 0.835359, 0.0, 0.3935726], [0.7179288, 0.0, 0.3935726, 0.0]],
interior_loop_length_count_mat_explicit: [[0.12663767, -0.09218068, -0.03926307, -0.07958286], [-0.09218068, 0.03552361, -0.03443626, -0.0016844398], [-0.03926307, -0.03443626, 0.003976357, 0.027582707], [-0.07958286, -0.0016844398, 0.027582707, 0.020128047]],
bulge_loop_0x1_length_counts: [0.009150381, -0.04190402, 0.016950816, -0.03667618],
interior_loop_1x1_length_count_mat: [[0.04903246, 0.10924888, 0.013425954, -0.09240063], [0.10924888, 0.07983476, 0.0050307703, 0.03929605], [0.013425954, 0.0050307703, 0.040781073, -0.07902155], [-0.09240063, 0.03929605, -0.07902155, 0.013872166]],
multi_loop_base_count: -0.30450365,
multi_loop_basepairing_count: -0.2813501,
multi_loop_accessible_baseunpairing_count: -0.1715042,
external_loop_accessible_basepairing_count: -0.23710607,
external_loop_accessible_baseunpairing_count: -0.18504995,
basepair_align_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.22586426], [0.0, 0.0, 0.019064197, 0.0], [0.0, -0.034546036, 0.0, -0.00808549], [-0.02631824, 0.0, -0.030726522, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.019064197], [0.0, 0.0, 0.2601551, 0.0], [0.0, -0.07323283, 0.0, 0.015945375], [-0.040399708, 0.0, -0.0009313092, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, -0.034546036], [0.0, 0.0, -0.07323283, 0.0], [0.0, 0.27139914, 0.0, 0.045052763], [-0.059358697, 0.0, 0.03151202, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, -0.00808549], [0.0, 0.0, 0.015945375, 0.0], [0.0, 0.045052763, 0.0, 0.15774933], [0.0030152854, 0.0, -0.044309106, 0.0]]], [[[0.0, 0.0, 0.0, -0.02631824], [0.0, 0.0, -0.040399708, 0.0], [0.0, -0.059358697, 0.0, 0.0030152854], [0.19706136, 0.0, 0.004082457, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, -0.030726522], [0.0, 0.0, -0.0009313092, 0.0], [0.0, 0.03151202, 0.0, -0.044309106], [0.004082457, 0.0, 0.06815988, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
loop_align_count_mat: [[0.89118636, -0.3151197, -0.20660992, -0.2050389], [-0.3151197, 0.6140849, -0.32944724, -0.17844164], [-0.20660992, -0.32944724, 0.599643, -0.279409], [-0.2050389, -0.17844164, -0.279409, 0.82097757]],
opening_gap_count: -3.136053,
extending_gap_count: -0.7198227,
hairpin_loop_length_counts_cumulative: [-0.2364569, -0.4394279, -0.59941417, -0.76747465, -0.99441266, -1.2370793, -1.3623539, -1.4959384, -1.6727269, -1.8363622, -1.9412974, -2.0695305, -2.2207885, -2.3506174, -2.4143867, -2.5421698, -2.6345603, -2.7184997, -2.8587523, -2.9476593, -2.993073, -3.0886874, -3.1457996, -3.1731844, -3.2308912, -3.269013, -3.2814631, -3.318758],
bulge_loop_length_counts_cumulative: [-0.3399576, -0.59285957, -0.81609833, -0.9464776, -1.0400212, -1.1386145, -1.2491211, -1.2967607, -1.4193016, -1.5140781, -1.5381181, -1.6012774, -1.6340712, -1.7159853, -1.7091538, -1.7335113, -1.7771825, -1.8290843, -1.8094006, -1.8409412, -1.8481838, -1.8641651, -1.8331642, -1.9034015, -1.9305975, -1.9187133, -1.9337721, -1.9026113, -1.9078758, -1.9430484],
interior_loop_length_counts_cumulative: [-0.17215727, -0.48962587, -0.76682335, -1.067966, -1.3060083, -1.4830564, -1.7161837, -1.9179845, -2.0698214, -2.2554214, -2.4250636, -2.599295, -2.72321, -2.8574715, -3.0020401, -3.203565, -3.3244572, -3.42612, -3.5127938, -3.5934725, -3.6397476, -3.739308, -3.8369803, -3.898003, -3.9742808, -3.9972615, -4.061592, -4.115016, -4.14438],
interior_loop_length_counts_symm_cumulative: [0.015577137, -0.06822123, -0.1584925, -0.22266525, -0.2937197, -0.3059198, -0.37817764, -0.4096338, -0.438639, -0.4947093, -0.54921985, -0.61088616, -0.63627696, -0.5773758, -0.5801587],
interior_loop_length_counts_asymm_cumulative: [-0.26209947, -0.32334593, -0.33761126, -0.31837732, -0.35816205, -0.37854448, -0.3446114, -0.3032234, -0.3174799, -0.29111767, -0.34189898, -0.37847432, -0.38977578, -0.37277737, -0.37194705, -0.37957248, -0.39359945, -0.4591822, -0.46826512, -0.53663677, -0.5748854, -0.5799931, -0.59402645, -0.5959138, -0.58079547, -0.59233207, -0.5886674, -0.5697153],
}
}
}