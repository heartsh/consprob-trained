impl FeatureCountSets {
pub fn load_trained_score_params() -> FeatureCountSets {
FeatureCountSets {
hairpin_loop_length_counts: [0.36103448, 6.968654, 1.2261529, 0.57220477, 0.6335132, 0.7356909, 1.4237032, 0.3610347, 0.34741074, 0.29291528, 0.21798371, 0.19073574, 0.054495927, 0.06130792, 0.013623979, 0.02043597, 0.040871944, 0.047683936, 0.08855588, 0.02043597, 0.006811989, 0.006811989, 0.0, 0.0, 0.0, 0.0, 0.0749319, 0.0],
bulge_loop_length_counts: [3.6989038, 0.24523167, 0.061307915, 0.08174389, 0.013623978, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
interior_loop_length_counts: [[8.658021, 1.8119841, 0.3746586, 0.6811989, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 2.9223397, 0.31335133, 0.17711176, 0.027247962, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.4168892, 0.061307915, 0.13623981, 0.02043597, 0.10217986, 0.006811989, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.24523167, 0.21798371, 0.04087194, 0.013623979, 0.0, 0.006811989, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.006811989, 0.04087194, 0.047683932, 0.006811989, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.013623979, 0.04087194, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.027247958, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.013623979, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
stack_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 5.00678], [0.0, 0.0, 4.1076427, 0.0], [0.0, 4.5776396, 0.0, 1.3078984], [4.611696, 0.0, 0.38828248, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 4.3664846], [0.0, 0.0, 7.6770353, 0.0], [0.0, 2.7861042, 0.0, 0.29972714], [4.5776396, 0.0, 3.0722075, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 6.3555484], [0.0, 0.0, 6.3964157, 0.0], [0.0, 7.6770353, 0.0, 2.0844631], [4.1076427, 0.0, 1.1307867, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.3896418], [0.0, 0.0, 1.1307867, 0.0], [0.0, 3.0722075, 0.0, 0.108991854], [0.38828248, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 3.181199], [0.0, 0.0, 6.3555484, 0.0], [0.0, 4.3664846, 0.0, 0.606264], [5.00678, 0.0, 1.3896418, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.606264], [0.0, 0.0, 2.0844631, 0.0], [0.0, 0.29972714, 0.0, 0.21798371], [1.3078984, 0.0, 0.108991854, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
hairpin_loop_terminal_mismatch_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.15667579, 0.08174389, 0.2247957], [0.006811989, 0.0, 0.17711176, 0.13623981], [0.29972735, 0.047683932, 0.006811989, 0.013623979], [0.60626495, 0.10217986, 0.9945447, 0.1498638]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.1498638, 1.5190704, 0.027247962, 0.3542226], [0.20435973, 0.5040866, 0.80380964, 0.027247962], [0.17029977, 0.8174335, 0.13623981, 0.08174389], [0.2724794, 0.39509442, 1.3147113, 0.21117172]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.115803845, 0.12942782, 0.02043597, 0.08174389], [1.0899124, 0.0, 0.12942782, 0.44277912], [0.38147148, 0.047683936, 0.0, 0.0], [0.2247957, 0.0, 0.08174389, 0.12261584]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.108991854, 0.006811989, 0.0749319, 0.0], [0.054495927, 0.0, 0.0, 0.0], [0.0, 0.0, 0.17711176, 0.0]]], [[[0.068119906, 0.04087194, 0.0, 0.1430518], [0.0, 0.054495927, 0.18392375, 0.013623979], [0.16348778, 0.013623979, 0.054495927, 0.04087194], [0.13623981, 0.04087194, 0.0, 0.047683936]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.04087194, 0.040871944, 0.02043597], [0.0, 0.0, 0.25204363, 0.0], [0.0, 0.0, 0.0, 0.16348778], [0.0, 0.0, 0.1430518, 0.013623979]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
interior_loop_terminal_mismatch_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.56539375, 0.6743837, 0.36103436, 0.4564018], [0.33378682, 0.91280097, 0.40190712, 0.4427776], [0.1430518, 0.53814554, 0.20435973, 0.108991854], [0.36784664, 0.5177089, 0.12942782, 0.45640284]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.49727353, 2.19346, 0.10217986, 0.37465852], [0.55858153, 0.36784694, 0.57901824, 0.37465918], [0.08855588, 0.30653912, 0.24523167, 0.5722056], [0.46321383, 1.0149802, 0.13623981, 0.26566756]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.45640197, 1.1580344, 0.15667579, 0.7288781], [0.41553, 1.1307858, 0.13623981, 0.51089704], [0.2247957, 0.6198886, 0.34741092, 0.4019065], [0.19073574, 1.0762887, 0.13623981, 0.76293844]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.26566762, 0.28610343, 0.2724796, 0.0749319], [0.1498638, 0.38147065, 0.04087194, 0.12942782], [0.20435973, 0.12261584, 0.08174389, 0.12942782], [0.1498638, 0.08174389, 0.006811989, 0.108991854]]], [[[0.08174389, 0.6335128, 0.25885558, 0.4427783], [0.5245212, 1.3896443, 0.59264165, 0.53814566], [0.068119906, 0.33378676, 0.108991854, 0.047683936], [1.0422282, 0.3337869, 0.21798371, 0.30653927]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.17029977, 0.013623979, 0.027247958], [0.6880085, 0.0, 0.1430518, 0.054495927], [0.040871944, 0.18392375, 0.2247957, 0.04087194], [0.24523167, 0.29972732, 0.0749319, 0.17711176]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
multi_loop_terminal_mismatch_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.027247962, 0.013623979, 0.0, 0.040871944], [0.0, 0.0, 0.0, 0.006811989], [0.0, 0.0, 0.0, 0.0], [0.027247962, 0.013623979, 0.0, 0.013623978]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.899178, 0.08174389, 0.0, 0.108991854], [0.89917797, 0.0, 0.0, 0.09536787], [0.0, 0.08174389, 0.0, 0.04087194], [0.08174389, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[1.0626645, 0.08174389, 0.0, 0.04087194], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.054495927, 0.08174389], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.013623978, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.006811989]]], [[[0.08174389, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.013623978, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.06130792]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.013623979], [0.98092115, 0.0, 0.0, 0.08174389], [0.0, 0.0, 0.0, 0.06130792]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
external_loop_left_terminal_mismatch_count_mat: [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.32016334, 0.80381143, 0.6335132, 0.17711176]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.08174389, 0.51771057, 0.2861036, 2.3024535], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [1.0967246, 0.64032507, 2.145772, 0.027247958], [0.0, 0.0, 0.0, 0.0], [0.068119906, 0.2792915, 0.2247957, 0.0]], [[0.34059915, 0.34059897, 0.23160769, 1.2057185], [0.0, 0.0, 0.0, 0.0], [0.108991854, 0.4291552, 0.0, 0.2997271], [0.0, 0.0, 0.0, 0.0]]],
external_loop_right_terminal_mismatch_count_mat: [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.19073574, 0.42915535, 0.80381143, 0.5108989]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.34196, 0.24523167, 1.4168874, 0.18392375], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.2247957, 1.0354168, 0.33378696, 2.3160732], [0.0, 0.0, 0.0, 0.0], [0.013623979, 0.020435967, 0.36784694, 0.17029977]], [[1.1512215, 0.17029977, 0.37465873, 0.42234254], [0.0, 0.0, 0.0, 0.0], [0.30653903, 0.0, 0.43596718, 0.09536787], [0.0, 0.0, 0.0, 0.0]]],
helix_end_count_mat: [[0.0, 0.0, 0.0, 11.151195], [0.0, 0.0, 17.472752, 0.0], [0.0, 16.661978, 0.0, 3.5149887], [10.524503, 0.0, 5.252009, 0.0]],
multi_loop_base_count: 1.3351412,
multi_loop_accessible_count: 3.7329736,
basepair_align_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 11.294271], [0.0, 0.0, 0.42234236, 0.0], [0.0, 2.0163424, 0.0, 0.39509472], [0.42915407, 0.0, 0.006811989, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.42234236], [0.0, 0.0, 16.498547, 0.0], [0.0, 0.7084423, 0.0, 0.08174389], [2.0163457, 0.0, 0.9400493, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 2.0163424], [0.0, 0.0, 0.7084423, 0.0], [0.0, 14.713885, 0.0, 0.783374], [0.3337869, 0.0, 0.10217986, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.39509472], [0.0, 0.0, 0.08174389, 0.0], [0.0, 0.783374, 0.0, 2.9904659], [0.1430518, 0.0, 0.006811989, 0.0]]], [[[0.0, 0.0, 0.0, 0.42915407], [0.0, 0.0, 2.0163457, 0.0], [0.0, 0.3337869, 0.0, 0.1430518], [13.065306, 0.0, 0.9264245, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.006811989], [0.0, 0.0, 0.9400493, 0.0], [0.0, 0.10217986, 0.0, 0.006811989], [0.9264245, 0.0, 2.806535, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
loop_align_count_mat: [[0.028322611, -9.500263, -11.860602, -9.605979], [-9.500263, -10.880817, -10.417539, -11.445974], [-11.860602, -10.417539, -15.308929, -9.94303], [-9.605979, -11.445974, -9.94303, -8.388397]],
opening_gap_count: -99.51992,
extending_gap_count: -35.117653,
}
}
}