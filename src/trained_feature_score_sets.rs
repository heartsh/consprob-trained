use FeatureCountSets;
impl FeatureCountSets {
pub fn load_trained_score_params() -> FeatureCountSets {
FeatureCountSets {
hairpin_loop_length_counts: [0.057760935, 0.05185896, 0.09417855, -0.022630023, -0.015321605, -0.048685323, -0.013343838, -0.0098652905, -0.008851506, -0.023239184, -0.08684095, 0.022710986, -0.0636837, -0.057418562, -0.019569779, 0.008658117, -0.056387883, -0.02737673, -0.0377343, -0.011508151, -0.07914423, -0.023403788, 0.019088339, 0.04319072, 0.0348732, 0.034414195, 0.038159534, -0.075794235],
bulge_loop_length_counts: [-0.2207239, -0.111865535, -0.061759315, -0.10301813, 0.0064083016, -0.0032981955, 0.0104999095, 0.008985601, -0.10210807, -0.0017715952, -0.03319289, -0.007913523, -0.0054057487, 0.010124518, 0.029579198, 0.024438174, 0.010155852, 0.0048222407, 0.042668022, 0.018157588, -0.006843651, -0.019944007, -0.00026867905, -0.0393528, -0.00042659894, -0.037889708, -0.016303314, 0.043957293, 0.02466779, -0.008818692],
interior_loop_length_counts: [-0.23973978, -0.24413848, -0.1543632, -0.21046813, -0.14394668, -0.105947495, -0.15921271, -0.12798119, -0.07692397, -0.08858087, -0.047617365, -0.08043618, -0.12774277, -0.08563445, -0.073231764, -0.06689093, -0.067631826, -0.07223749, -0.08325925, -0.0071027763, 0.002444444, -0.07436635, -0.05168802, -0.029257756, -0.04113651, -0.04998531, -0.0006498958, 0.0073316945, 0.011217155],
interior_loop_length_counts_symm: [-0.09433622, -0.104955554, -0.07872079, -0.029324582, 0.0006436646, -0.04949967, 0.013900589, 0.00060447014, -0.012624756, -0.022049556, -0.00904001, 0.020083634, -0.017473318, -0.008806547, -0.008559777],
interior_loop_length_counts_asymm: [-0.15870854, -0.035956215, -0.031729437, -0.030866025, -0.013956525, 0.0005054241, -0.01799251, -0.0020883945, -0.0068028, -0.030089734, 0.020092368, -0.036822654, -0.029489992, 0.015760615, 0.040804047, 0.03946017, -0.017793912, -0.015333299, 0.021945069, 0.020103838, -0.05869161, -0.00414987, 0.029057026, 0.04525603, 0.019712588, 0.03694024, 0.008607347, 0.012048109],
stack_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.17461577], [0.0, 0.0, 0.1405092, 0.0], [0.0, 0.12835084, 0.0, -0.0013466506], [0.057206172, 0.0, 0.023486935, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.16653001], [0.0, 0.0, 0.18977822, 0.0], [0.0, 0.08504284, 0.0, 0.005775514], [0.12835084, 0.0, 0.07699883, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.13552593], [0.0, 0.0, 0.11609387, 0.0], [0.0, 0.18977822, 0.0, 0.04218901], [0.1405092, 0.0, 0.05050427, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.08564787], [0.0, 0.0, 0.05050427, 0.0], [0.0, 0.07699883, 0.0, 0.0016139863], [0.023486935, 0.0, -0.03007986, 0.0]]], [[[0.0, 0.0, 0.0, 0.103942655], [0.0, 0.0, 0.13552593, 0.0], [0.0, 0.16653001, 0.0, 0.054898996], [0.17461577, 0.0, 0.08564787, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.054898996], [0.0, 0.0, 0.04218901, 0.0], [0.0, 0.005775514, 0.0, 0.07036949], [-0.0013466506, 0.0, 0.0016139863, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
terminal_mismatch_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.01285177, -0.0052453442, -0.00502408, -0.04477154], [0.016550088, 0.022106478, -0.07674669, -0.0024416568], [-0.020597324, -0.057375632, 0.0101912245, -0.004192998], [-0.09713409, -0.0024322304, -0.021261437, 0.015973115]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0259307, -0.008536109, 0.010020856, -0.0933749], [-0.024434384, 0.014483838, -0.06698519, -0.0020118428], [-0.013811278, -0.1074403, -0.060925614, -0.05589019], [-0.03887704, 0.00226542, -0.05047797, 0.014422795]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.0051326966, 0.013724132, -0.02405018, -0.07542456], [0.020845074, -0.026610624, -0.07967902, 0.036078494], [0.032461822, -0.062481683, 0.06452991, -0.0050924793], [-0.031849064, -0.008060332, 0.029676424, 0.047787994]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.03371827, 0.01486843, -0.011028113, -0.020882787], [-0.03526594, 0.0240353, -0.015734512, -0.003026048], [-0.00530177, -0.012505309, 0.05139446, -0.028953906], [-0.042914473, -0.020348875, 0.036663834, -0.021599378]]], [[[-0.0036663045, 0.008106228, 0.060305014, 0.012175196], [0.025044803, -0.022593934, -0.019163623, 0.06353807], [-0.01700167, -0.064096265, -0.014059031, 0.014176849], [-0.0913056, 0.014861476, 0.022779806, -0.035832945]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[-0.059514202, -0.0029676568, -0.02105531, -0.05079147], [-0.020396512, 0.043218482, -0.02167259, -0.06553737], [0.018535772, -0.030086694, -0.00011704021, -0.05862984], [-0.035883565, 0.026076412, -0.0059165386, -0.04273728]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
left_dangle_count_mat: [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.014875946, 0.021414591, -0.04946965, -0.001647841]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.024597328, -0.04724779, 0.019681636, 0.0066382373], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [-0.045861665, -0.037635233, -0.030459808, -0.083557725], [0.0, 0.0, 0.0, 0.0], [0.020063842, -0.022489587, 0.035275996, 0.03896135]], [[-0.023953691, -0.034305397, -0.031889036, -0.05484068], [0.0, 0.0, 0.0, 0.0], [0.0017290567, 0.032948617, 0.027184919, 0.022138936], [0.0, 0.0, 0.0, 0.0]]],
right_dangle_count_mat: [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.022245685, -0.026142275, 0.019890005, -0.024398332]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [-0.020780751, 0.0357643, 0.031663477, -0.039604165], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [-0.024576038, -0.009007874, -0.03963554, -0.0067924103], [0.0, 0.0, 0.0, 0.0], [-0.035273924, -0.005130597, 0.023799503, 0.0060530542]], [[-0.044822965, -0.022176847, 0.029135318, 0.024347363], [0.0, 0.0, 0.0, 0.0], [-0.010375209, 0.06852591, -0.023870539, -0.0101137245], [0.0, 0.0, 0.0, 0.0]]],
helix_end_count_mat: [[0.0, 0.0, 0.0, -0.32235515], [0.0, 0.0, -0.30188283, 0.0], [0.0, -0.3394243, 0.0, -0.061405614], [-0.3039864, 0.0, -0.088048406, 0.0]],
base_pair_count_mat: [[0.0, 0.0, 0.0, 0.2909377], [0.0, 0.0, 0.43177065, 0.0], [0.0, 0.43177065, 0.0, 0.105312124], [0.2909377, 0.0, 0.105312124, 0.0]],
interior_loop_length_count_mat_explicit: [[0.03277813, -0.03276185, 0.011880314, -0.0712185], [-0.03276185, -0.013809341, -0.050843984, 0.027022358], [0.011880314, -0.050843984, -0.022912866, 0.02185265], [-0.0712185, 0.027022358, 0.02185265, -0.020817585]],
bulge_loop_0x1_length_counts: [0.023462437, -0.04028313, -0.041419744, -0.058722045],
interior_loop_1x1_length_count_mat: [[0.06538288, 0.045305677, -0.04364164, -0.07940936], [0.045305677, 0.0065823714, -0.032295328, -0.014473822], [-0.04364164, -0.032295328, -0.011434304, -0.031128805], [-0.07940936, -0.014473822, -0.031128805, 0.03695105]],
multi_loop_base_count: -0.14253189,
multi_loop_basepairing_count: -0.18328808,
multi_loop_accessible_baseunpairing_count: -0.12300065,
external_loop_accessible_basepairing_count: -0.11003629,
external_loop_accessible_baseunpairing_count: -0.12762146,
basepair_align_count_mat: [[[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.10710201], [0.0, 0.0, -0.028404918, 0.0], [0.0, 0.019268628, 0.0, 0.026120186], [-0.04881639, 0.0, 0.039925065, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, -0.028404918], [0.0, 0.0, 0.11624722, 0.0], [0.0, 0.0020673743, 0.0, 0.031046309], [-0.03473778, 0.0, -0.0013739138, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.019268628], [0.0, 0.0, 0.0020673743, 0.0], [0.0, 0.09731211, 0.0, 0.0056635104], [0.014321412, 0.0, -0.022331716, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.026120186], [0.0, 0.0, 0.031046309, 0.0], [0.0, 0.0056635104, 0.0, 0.030545823], [-0.022670599, 0.0, 0.044524428, 0.0]]], [[[0.0, 0.0, 0.0, -0.04881639], [0.0, 0.0, -0.03473778, 0.0], [0.0, 0.014321412, 0.0, -0.022670599], [0.064171895, 0.0, 0.033059802, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.039925065], [0.0, 0.0, -0.0013739138, 0.0], [0.0, -0.022331716, 0.0, 0.044524428], [0.033059802, 0.0, 0.04028291, 0.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]],
loop_align_count_mat: [[0.12487308, -0.23112267, -0.22244075, -0.22230342], [-0.23112267, -0.0022784504, -0.21365009, -0.23595576], [-0.22244075, -0.21365009, -0.045187104, -0.2259366], [-0.22230342, -0.23595576, -0.2259366, 0.037838724]],
match_2_match_count: 4.1107993,
match_2_insert_count: -0.508483,
insert_extend_count: 0.02448946,
insert_switch_count: -0.49758843,
insert_counts: [-0.5088488, -0.4124869, -0.4390074, -0.5661071],
align_count_mat: [[0.4052396, -0.18264636, -0.08349873, -0.18130976], [-0.18264636, 0.3408632, -0.18565018, -0.07288644], [-0.08349873, -0.18565018, 0.43643212, -0.17828625], [-0.18130976, -0.07288644, -0.17828625, 0.3952845]],
hairpin_loop_length_counts_cumulative: [0.057760935, 0.1096199, 0.20379844, 0.18116842, 0.16584682, 0.1171615, 0.10381766, 0.093952365, 0.08510086, 0.061861675, -0.024979275, -0.0022682883, -0.06595199, -0.12337055, -0.14294033, -0.13428222, -0.1906701, -0.21804683, -0.2557811, -0.26728925, -0.3464335, -0.36983728, -0.35074896, -0.30755824, -0.27268505, -0.23827085, -0.20011131, -0.27590555],
bulge_loop_length_counts_cumulative: [-0.2207239, -0.33258945, -0.39434877, -0.4973669, -0.4909586, -0.4942568, -0.4837569, -0.4747713, -0.5768794, -0.57865095, -0.6118438, -0.61975735, -0.6251631, -0.6150386, -0.58545935, -0.56102115, -0.5508653, -0.54604304, -0.503375, -0.4852174, -0.49206105, -0.51200503, -0.5122737, -0.5516265, -0.5520531, -0.5899428, -0.6062461, -0.5622888, -0.537621, -0.5464397],
interior_loop_length_counts_cumulative: [-0.23973978, -0.48387825, -0.63824147, -0.8487096, -0.99265623, -1.0986037, -1.2578164, -1.3857976, -1.4627216, -1.5513024, -1.5989197, -1.679356, -1.8070987, -1.8927332, -1.965965, -2.032856, -2.1004877, -2.1727252, -2.2559845, -2.2630873, -2.2606428, -2.335009, -2.386697, -2.4159548, -2.4570913, -2.5070767, -2.5077267, -2.500395, -2.489178],
interior_loop_length_counts_symm_cumulative: [-0.09433622, -0.19929177, -0.27801257, -0.30733716, -0.3066935, -0.35619316, -0.34229258, -0.3416881, -0.35431287, -0.3763624, -0.3854024, -0.36531878, -0.3827921, -0.39159864, -0.4001584],
interior_loop_length_counts_asymm_cumulative: [-0.15870854, -0.19466476, -0.2263942, -0.25726023, -0.27121675, -0.27071133, -0.28870383, -0.29079223, -0.29759502, -0.32768476, -0.3075924, -0.34441504, -0.37390503, -0.3581444, -0.31734034, -0.27788016, -0.2956741, -0.31100738, -0.28906232, -0.26895848, -0.3276501, -0.33179998, -0.30274296, -0.25748694, -0.23777436, -0.20083413, -0.19222678, -0.18017867],
}
}
}