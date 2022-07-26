use FeatureCountSets;
impl FeatureCountSets {
pub fn load_trained_score_params() -> FeatureCountSets {
FeatureCountSets {
hairpin_loop_length_counts: [-6.006627, -3.120398, 0.4043042, 2.1920433, 1.9092307, -0.6031472, -0.099120475, 0.56268734, -0.73497957, -0.18412066, -0.31293157, -0.040486105, -0.9363807, -0.040438738, -0.11202074, 0.18197873, -0.10014634, 0.16280359, -0.08844706, -0.349931, -0.11362859, -0.29304942, -0.3416511, -0.19373316, -0.052151494, -0.040632863, 0.04615468, 0.066556215, 0.096548036, 0.16696586, 0.23273082],
bulge_loop_length_counts: [-2.395066, -0.89329654, -0.90795356, -0.84008497, -0.4352768, -0.56853735, 0.20029218, 0.75417817, -0.6039324, -0.72014725, -0.5140325, -0.36182866, -0.26176023, -0.15968704, -0.0865223, -0.031195963, -0.0110956095, 0.029899772, 0.047485404, -0.04306392, -0.017990896, -0.078075334, -0.071049325, -0.057734627, -0.046374045, -0.03562938, -0.02677873, -0.018211728, -0.010544839, -0.005154814],
interior_loop_length_counts: [-0.40164247, -0.34637794, -0.38784453, -0.314443, -0.2620114, -0.060549982, -0.052982233, -0.009183305, -0.18612912, -0.27269053, -0.35358283, -0.2993309, -0.04301611, -0.12813044, -0.05055169, -0.08879599, -0.013627694, 0.01735085, 0.025038602, -0.07097377, -0.13581136, -0.1444065, -0.069028154, -0.08037228, -0.05300111, -0.0431463, 0.0019087429, 0.0049535404, 0.006627036],
interior_loop_length_counts_symm: [-0.5140542, -0.37340242, -0.258586, -0.23613667, 0.14246368, -0.65775484, -0.30323336, -0.0314552, -0.35271904, -0.21674927, -0.12339443, -0.15562315, -0.08568161, -0.04606582, -0.022369001],
interior_loop_length_counts_asymm: [-2.1108112, -0.55391014, -0.5784694, -0.6149126, -0.30688888, -0.11643609, -0.21111901, -0.31494406, -0.3154598, -0.09062696, -0.22029385, -0.14084966, -0.21639782, -0.17266977, -0.15600075, -0.104120225, -0.069686785, -0.04106719, -0.015713017, 0.013838025, 0.04135444, 0.035977356, 0.028262235, 0.016472137, 0.025623063, 0.033619344, 0.039717425, -0.0025451537],
stack_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.12712623], [0.0, 0.0, 0.42393214, 0.0], [0.0, 0.6858303, 0.0, -0.10182096], [0.23734508, 0.0, 0.15727337, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.47019526], [0.0, 0.0, 0.8265241, 0.0], [0.0, 0.47395957, 0.0, -0.18014432], [0.6858303, 0.0, 0.47857124, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.5292416], [0.0, 0.0, 0.49269152, 0.0], [0.0, 0.8265241, 0.0, 0.21654566], [0.42393214, 0.0, 0.4829103, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, -0.047750466], [0.0, 0.0, 0.4829103, 0.0], [0.0, 0.47857124, 0.0, 0.18422672], [0.15727337, 0.0, -0.287462, 0.0]]], [[[0.0, 0.0, 0.0, 0.38894096], [0.0, 0.0, 0.5292416, 0.0], [0.0, 0.47019526, 0.0, -0.115009345], [0.12712623, 0.0, -0.047750466, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, -0.115009345], [0.0, 0.0, 0.21654566, 0.0], [0.0, -0.18014432, 0.0, 0.11920485], [-0.10182096, 0.0, 0.18422672, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
terminal_mismatch_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.18511823, -0.11619542, -0.44528487, -0.61279416], [0.004751757, 0.08354787, -0.22263867, -0.3962524], [0.5204676, -0.34691134, -0.40505636, -0.77110124], [-0.016115135, 0.26832595, -0.09099928, 0.3356964]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.08080664, -0.2522797, -0.6703052, -0.38093793], [0.1136039, -0.1706737, -0.2145033, -0.46068516], [0.8494148, -0.93068326, -0.32868662, -0.77600765], [-0.23722793, -0.039323043, -0.43029624, -0.24277024]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.17093663, -0.09118434, -0.25133178, -0.8475693], [0.04784466, -0.24436891, -0.20666285, -0.18778107], [0.6545556, -0.7807701, 0.20145829, -0.44321445], [-0.17244373, 0.2866708, -0.016227894, 0.67500204]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.48662266, 0.11313345, 0.3632672, -0.61847055], [0.34588996, 0.030720547, -0.3774431, -0.031928305], [0.49531242, -0.28144524, -0.26957172, -0.065975785], [-0.42970335, -0.09293557, -0.31252298, -0.22589333]]], [[[0.009940903, -0.39201918, 0.056065336, -0.12281441], [-0.06278753, -0.31651294, 0.003971037, -0.421427], [0.5446472, -0.20535797, -0.19664657, -0.47146586], [-0.17519455, 0.16507365, -0.49858078, 0.13313538]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.12092549, 0.19897501, 0.046365693, 0.32381436], [0.120358706, -0.18531002, -0.04295296, -0.61524856], [0.7545477, -0.31415406, 0.15713792, -0.51367956], [-0.29329962, 0.13740735, -0.05355261, 0.029824553]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
left_dangle_count_mat: [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.12617387, 0.04406562, -0.02508946, 0.008032313]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.04179423, -0.022404894, 0.099273086, -0.15208802], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [-0.1819971, 0.028365288, 0.13211918, -0.16496964], [0.0, 0.0, 0.0, 0.0], [-0.06446925, -0.04231648, 0.028389951, -0.042250354]], [[-0.03069119, -0.0061478727, -0.11761708, -0.012331839], [0.0, 0.0, 0.0, 0.0], [-0.080577254, 0.0013516556, 0.10206434, -0.09280465], [0.0, 0.0, 0.0, 0.0]]],
right_dangle_count_mat: [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.032124028, -0.090482675, -0.07327536, -0.017876646]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.19939105, -0.114104636, -0.07117438, -0.21549156], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.014396969, 0.0035341296, -0.010633412, -0.25859186], [0.0, 0.0, 0.0, 0.0], [-0.044089198, -0.07325413, 0.013146891, -0.05642803]], [[-0.16374944, 0.06895142, -0.08762269, -0.05397557], [0.0, 0.0, 0.0, 0.0], [0.040636547, -0.008681034, -0.03857711, -0.08563789], [0.0, 0.0, 0.0, 0.0]]],
helix_end_count_mat: [[0.0, 0.0, 0.0, -0.9549985], [0.0, 0.0, -0.5779577, 0.0], [0.0, -0.8265224, 0.0, -1.0355016], [-0.907049, 0.0, -0.3714595, 0.0]],
base_pair_count_mat: [[0.0, 0.0, 0.0, 0.54864794], [0.0, 0.0, 1.4718258, 0.0], [0.0, 1.4718258, 0.0, -0.015329712], [0.54864794, 0.0, -0.015329712, 0.0]],
interior_loop_length_count_mat_explicit: [[-0.15486926, 0.029138537, -0.17150477, -0.22957489], [0.029138537, -0.11859791, -0.0736876, 0.27826715], [-0.17150477, -0.0736876, -0.026675098, 0.31116578], [-0.22957489, 0.27826715, 0.31116578, -0.32209137]],
bulge_loop_0x1_length_counts: [-0.122133315, -0.069878995, 0.011801979, -0.0031039414],
interior_loop_1x1_length_count_mat: [[0.29299802, 0.092197105, -0.36412144, -0.20036586], [0.092197105, -0.15855941, 0.42369866, 0.13715304], [-0.36412144, 0.42369866, -0.11882742, -0.41618907], [-0.20036586, 0.13715304, -0.41618907, 0.14685473]],
multi_loop_base_count: -1.2033341,
multi_loop_basepairing_count: -0.9294953,
multi_loop_accessible_baseunpairing_count: -0.20910744,
external_loop_accessible_basepairing_count: -0.005182006,
external_loop_accessible_baseunpairing_count: -0.21943137,
match_2_match_count: 2.6281595,
match_2_insert_count: 0.16329537,
insert_extend_count: 1.0116884,
init_match_count: 0.3975269,
init_insert_count: -0.35034367,
insert_counts: [-0.009871484, -0.08378512, -0.07389463, -0.024384325],
align_count_mat: [[0.5177172, -0.40225968, -0.24346305, -0.31438884], [-0.40225968, 0.65925807, -0.32091177, -0.13153742], [-0.24346305, -0.32091177, 0.6573448, -0.3488858], [-0.31438884, -0.13153742, -0.3488858, 0.4557519]],
hairpin_loop_length_counts_cumulative: [-6.006627, -9.127026, -8.722721, -6.530678, -4.621447, -5.224594, -5.3237147, -4.7610273, -5.496007, -5.6801276, -5.993059, -6.0335455, -6.9699264, -7.010365, -7.122386, -6.9404073, -7.0405536, -6.87775, -6.966197, -7.316128, -7.429756, -7.7228055, -8.064457, -8.25819, -8.310342, -8.350975, -8.30482, -8.238264, -8.141716, -7.97475, -7.742019],
bulge_loop_length_counts_cumulative: [-2.395066, -3.2883625, -4.1963162, -5.0364013, -5.4716783, -6.0402155, -5.8399234, -5.0857453, -5.6896777, -6.409825, -6.923857, -7.285686, -7.5474463, -7.7071333, -7.7936554, -7.8248515, -7.835947, -7.8060474, -7.758562, -7.801626, -7.8196173, -7.8976927, -7.968742, -8.026477, -8.072851, -8.10848, -8.13526, -8.153471, -8.164016, -8.16917],
interior_loop_length_counts_cumulative: [-0.40164247, -0.7480204, -1.135865, -1.450308, -1.7123194, -1.7728693, -1.8258516, -1.8350348, -2.021164, -2.2938545, -2.6474373, -2.9467683, -2.9897845, -3.117915, -3.1684666, -3.2572625, -3.2708902, -3.2535393, -3.2285006, -3.2994745, -3.4352858, -3.5796924, -3.6487205, -3.7290928, -3.782094, -3.8252404, -3.8233316, -3.818378, -3.811751],
interior_loop_length_counts_symm_cumulative: [-0.5140542, -0.8874566, -1.1460426, -1.3821793, -1.2397156, -1.8974705, -2.2007039, -2.2321591, -2.5848782, -2.8016274, -2.925022, -3.080645, -3.1663268, -3.2123926, -3.2347615],
interior_loop_length_counts_asymm_cumulative: [-2.1108112, -2.6647215, -3.2431908, -3.8581033, -4.1649923, -4.2814283, -4.4925475, -4.807492, -5.1229515, -5.2135787, -5.4338727, -5.5747223, -5.79112, -5.96379, -6.1197906, -6.223911, -6.2935977, -6.334665, -6.350378, -6.33654, -6.2951856, -6.259208, -6.230946, -6.2144737, -6.188851, -6.1552315, -6.1155143, -6.1180596],
}
}
}