extern crate bio;
extern crate consprob;
extern crate my_bfgs as bfgs;
extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;
pub mod trained_feature_score_sets;
pub mod trained_feature_score_sets_random_init;

pub use bfgs::bfgs;
pub use bio::io::fasta::*;
pub use bio::utils::*;
pub use consprob::*;
pub use ndarray::prelude::*;
pub use ndarray::Array;
pub use ndarray_rand::rand_distr::{Distribution, Normal};
pub use ndarray_rand::RandomExt;
pub use rand::thread_rng;
pub use std::f32::INFINITY;
pub use std::fs::{read_dir, DirEntry};
pub use std::io::stdout;

pub type SparsePosMat<T> = HashSet<PosPair<T>>;
pub type RealBpScoreParamSetPair<T> = (BpScoreParamSets<T>, BpScoreParamSets<T>);
pub type BpScoreParamSetPair<'a, T> = (&'a BpScoreParamSets<T>, &'a BpScoreParamSets<T>);
pub type BpScores<T> = HashMap<PosPair<T>, FeatureCount>;
pub type TwoloopScores<T> = HashMap<PosQuadruple<T>, FeatureCount>;
pub type Regularizers = Array1<Regularizer>;
pub type Regularizer = FeatureCount;
pub type BfgsFeatureCounts = Array1<BfgsFeatureCount>;
pub type BfgsFeatureCount = f64;
pub type FeatureCounts = Array1<FeatureCount>;
pub type TrainData<T> = Vec<TrainDatum<T>>;
pub type AlignCountMat = HelixEndCountMat;
pub type InsertCounts = [FeatureCount; NUM_OF_BASES];
pub type RealSeqPair = (Seq, Seq);
pub type LoopStruct = HashMap<(usize, usize), Vec<(usize, usize)>>;

#[derive(Clone)]
pub struct BpScoreParamSets<T> {
  pub hairpin_loop_scores: BpScores<T>,
  pub twoloop_scores: TwoloopScores<T>,
  pub multi_loop_closing_bp_scores: BpScores<T>,
  pub multi_loop_accessible_bp_scores: BpScores<T>,
  pub external_loop_accessible_bp_scores: BpScores<T>,
}

#[derive(Clone)]
pub struct TrainDatum<T> {
  pub seq_pair: RealSeqPair,
  pub seq_pair_with_gaps: RealSeqPair,
  pub observed_feature_count_sets: FeatureCountSets,
  pub expected_feature_count_sets: FeatureCountSets,
  pub bpp_mat_pair: SparseProbMatPair<T>,
  pub max_bp_span_pair: (T, T),
  pub part_func: Prob,
  pub forward_pos_pair_mat_set: PosPairMatSet<T>,
  pub backward_pos_pair_mat_set: PosPairMatSet<T>,
  pub pos_quadruple_mat_with_len_pairs: PosQuadrupleMatWithLenPairs<T>,
  pub matchable_pos_sets_1: SparsePosSets<T>,
  pub matchable_pos_sets_2: SparsePosSets<T>,
  pub bp_score_param_set_pair: RealBpScoreParamSetPair<T>,
  pub align_prob_mat: SparseProbMat<T>,
  pub align: PairAlign<T>,
  pub acc: FeatureCount,
}

#[derive(Clone)]
pub struct PairAlign<T> {
  pub aligned_pos_pair_mat: SparsePosMat<T>,
  pub inserted_poss: SparsePoss<T>,
  pub deleted_poss: SparsePoss<T>,
}

#[derive(Clone, Debug)]
pub struct FeatureCountSets {
  // The CONTRAfold model.
  pub hairpin_loop_length_counts: HairpinLoopLengthCounts,
  pub bulge_loop_length_counts: BulgeLoopLengthCounts,
  pub interior_loop_length_counts: InteriorLoopLengthCounts,
  pub interior_loop_length_counts_symm: InteriorLoopLengthCountsSymm,
  pub interior_loop_length_counts_asymm: InteriorLoopLengthCountsAsymm,
  pub stack_count_mat: StackCountMat,
  pub terminal_mismatch_count_mat: TerminalMismatchCount4dMat,
  pub left_dangle_count_mat: DangleCount3dMat,
  pub right_dangle_count_mat: DangleCount3dMat,
  pub helix_end_count_mat: HelixEndCountMat,
  pub base_pair_count_mat: BasePairCountMat,
  pub interior_loop_length_count_mat_explicit: InteriorLoopLengthCountMatExplicit,
  pub bulge_loop_0x1_length_counts: BulgeLoop0x1LengthCounts,
  pub interior_loop_1x1_length_count_mat: InteriorLoop1x1LengthCountMat,
  pub multi_loop_base_count: FeatureCount,
  pub multi_loop_basepairing_count: FeatureCount,
  pub multi_loop_accessible_baseunpairing_count: FeatureCount,
  pub external_loop_accessible_basepairing_count: FeatureCount,
  pub external_loop_accessible_baseunpairing_count: FeatureCount,
  // The CONTRAlign model.
  pub match_2_match_count: FeatureCount,
  pub match_2_insert_count: FeatureCount,
  pub insert_extend_count: FeatureCount,
  pub init_match_count: FeatureCount,
  pub init_insert_count: FeatureCount,
  pub insert_counts: InsertCounts,
  pub align_count_mat: AlignCountMat,
  // The cumulative parameters of the CONTRAfold model.
  pub hairpin_loop_length_counts_cumulative: HairpinLoopLengthCounts,
  pub bulge_loop_length_counts_cumulative: BulgeLoopLengthCounts,
  pub interior_loop_length_counts_cumulative: InteriorLoopLengthCounts,
  pub interior_loop_length_counts_symm_cumulative: InteriorLoopLengthCountsSymm,
  pub interior_loop_length_counts_asymm_cumulative: InteriorLoopLengthCountsAsymm,
}
pub type FeatureCountMat = Vec<Vec<FeatureCount>>;
pub struct InsertScoreRangeSets {
  pub insert_scores_range: FeatureCountMat,
  pub insert_scores_range_4_el: FeatureCountMat,
  pub insert_scores_range_4_ml: FeatureCountMat,
  pub insert_scores_range_2: FeatureCountMat,
  pub insert_scores_range_4_el_2: FeatureCountMat,
  pub insert_scores_range_4_ml_2: FeatureCountMat,
}

impl<T: HashIndex> BpScoreParamSets<T> {
  pub fn new() -> BpScoreParamSets<T> {
    BpScoreParamSets {
      hairpin_loop_scores: BpScores::<T>::default(),
      twoloop_scores: TwoloopScores::<T>::default(),
      multi_loop_closing_bp_scores: BpScores::<T>::default(),
      multi_loop_accessible_bp_scores: BpScores::<T>::default(),
      external_loop_accessible_bp_scores: BpScores::<T>::default(),
    }
  }

  pub fn set_curr_params(
    feature_score_sets: &FeatureCountSets,
    seq: SeqSlice,
    bpp_mat: &SparseProbMat<T>,
  ) -> BpScoreParamSets<T> {
    let seq_len = seq.len();
    let mut bp_score_param_sets = BpScoreParamSets::<T>::new();
    for pos_pair in bpp_mat.keys() {
      let long_pos_pair = (
        pos_pair.0.to_usize().unwrap(),
        pos_pair.1.to_usize().unwrap(),
      );
      if long_pos_pair.1 - long_pos_pair.0 - 1 <= CONSPROB_MAX_TWOLOOP_LEN {
        let hairpin_loop_score = get_hl_fe_trained(feature_score_sets, seq, &long_pos_pair);
        bp_score_param_sets
          .hairpin_loop_scores
          .insert(*pos_pair, hairpin_loop_score);
      }
      let multi_loop_closing_basepairing_score = feature_score_sets.multi_loop_base_count
        + feature_score_sets.multi_loop_basepairing_count
        + get_junction_fe_multi_trained(feature_score_sets, seq, &long_pos_pair, seq_len);
      bp_score_param_sets
        .multi_loop_closing_bp_scores
        .insert(*pos_pair, multi_loop_closing_basepairing_score);
      let base_pair = (seq[long_pos_pair.0], seq[long_pos_pair.1]);
      let fe_multi = get_junction_fe_multi_trained(
        feature_score_sets,
        seq,
        &(long_pos_pair.1, long_pos_pair.0),
        seq_len,
      ) + feature_score_sets.base_pair_count_mat[base_pair.0][base_pair.1];
      let multi_loop_accessible_basepairing_score =
        fe_multi + feature_score_sets.multi_loop_basepairing_count;
      bp_score_param_sets
        .multi_loop_accessible_bp_scores
        .insert(*pos_pair, multi_loop_accessible_basepairing_score);
      let external_loop_accessible_basepairing_score =
        fe_multi + feature_score_sets.external_loop_accessible_basepairing_count;
      bp_score_param_sets
        .external_loop_accessible_bp_scores
        .insert(*pos_pair, external_loop_accessible_basepairing_score);
      for pos_pair_2 in bpp_mat.keys() {
        if !(pos_pair_2.0 < pos_pair.0 && pos_pair.1 < pos_pair_2.1) {
          continue;
        }
        let long_pos_pair_2 = (
          pos_pair_2.0.to_usize().unwrap(),
          pos_pair_2.1.to_usize().unwrap(),
        );
        if long_pos_pair.0 - long_pos_pair_2.0 - 1 + long_pos_pair_2.1 - long_pos_pair.1 - 1
          > CONSPROB_MAX_TWOLOOP_LEN
        {
          continue;
        }
        let twoloop_score =
          get_consprob_twoloop_score(feature_score_sets, seq, &long_pos_pair_2, &long_pos_pair);
        bp_score_param_sets.twoloop_scores.insert(
          (pos_pair_2.0, pos_pair_2.1, pos_pair.0, pos_pair.1),
          twoloop_score,
        );
      }
    }
    bp_score_param_sets
  }
}

impl FeatureCountSets {
  pub fn new(init_val: FeatureCount) -> FeatureCountSets {
    let init_vals = [init_val; NUM_OF_BASES];
    let twod_mat = [init_vals; NUM_OF_BASES];
    let threed_mat = [twod_mat; NUM_OF_BASES];
    let fourd_mat = [threed_mat; NUM_OF_BASES];
    FeatureCountSets {
      // The CONTRAfold model.
      hairpin_loop_length_counts: [init_val; CONSPROB_MAX_HAIRPIN_LOOP_LEN + 1],
      bulge_loop_length_counts: [init_val; CONSPROB_MAX_TWOLOOP_LEN],
      interior_loop_length_counts: [init_val; CONSPROB_MAX_TWOLOOP_LEN - 1],
      interior_loop_length_counts_symm: [init_val; CONSPROB_MAX_INTERIOR_LOOP_LEN_SYMM],
      interior_loop_length_counts_asymm: [init_val; CONSPROB_MAX_INTERIOR_LOOP_LEN_ASYMM],
      stack_count_mat: fourd_mat,
      terminal_mismatch_count_mat: fourd_mat,
      left_dangle_count_mat: threed_mat,
      right_dangle_count_mat: threed_mat,
      helix_end_count_mat: twod_mat,
      base_pair_count_mat: twod_mat,
      interior_loop_length_count_mat_explicit: [[init_val; CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT];
        CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT],
      bulge_loop_0x1_length_counts: [init_val; NUM_OF_BASES],
      interior_loop_1x1_length_count_mat: [[init_val; NUM_OF_BASES]; NUM_OF_BASES],
      multi_loop_base_count: init_val,
      multi_loop_basepairing_count: init_val,
      multi_loop_accessible_baseunpairing_count: init_val,
      external_loop_accessible_basepairing_count: init_val,
      external_loop_accessible_baseunpairing_count: init_val,
      // The CONTRAlign model.
      match_2_match_count: init_val,
      match_2_insert_count: init_val,
      init_match_count: init_val,
      insert_extend_count: init_val,
      init_insert_count: init_val,
      insert_counts: init_vals,
      align_count_mat: twod_mat,
      // The cumulative parameters of the CONTRAfold model.
      hairpin_loop_length_counts_cumulative: [init_val; CONSPROB_MAX_HAIRPIN_LOOP_LEN + 1],
      bulge_loop_length_counts_cumulative: [init_val; CONSPROB_MAX_TWOLOOP_LEN],
      interior_loop_length_counts_cumulative: [init_val; CONSPROB_MAX_TWOLOOP_LEN - 1],
      interior_loop_length_counts_symm_cumulative: [init_val; CONSPROB_MAX_INTERIOR_LOOP_LEN_SYMM],
      interior_loop_length_counts_asymm_cumulative: [init_val;
        CONSPROB_MAX_INTERIOR_LOOP_LEN_ASYMM],
    }
  }

  pub fn get_len(&self) -> usize {
    let mut sub_len = 0;
    sub_len += self.hairpin_loop_length_counts.len();
    sub_len += self.bulge_loop_length_counts.len();
    sub_len += self.interior_loop_length_counts.len();
    sub_len += self.interior_loop_length_counts_symm.len();
    sub_len += self.interior_loop_length_counts_asymm.len();
    let len = self.stack_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for k in 0..len {
          for l in 0..len {
            if !is_canonical(&(k, l)) {
              continue;
            }
            let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
            if ((i, j), (k, l)) != dict_min_stack {
              continue;
            }
            sub_len += 1;
          }
        }
      }
    }
    let len = self.terminal_mismatch_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for _ in 0..len {
          for _ in 0..len {
            sub_len += 1;
          }
        }
      }
    }
    let len = self.left_dangle_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for _ in 0..len {
          sub_len += 1;
        }
      }
    }
    let len = self.right_dangle_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for _ in 0..len {
          sub_len += 1;
        }
      }
    }
    let len = self.helix_end_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        sub_len += 1;
      }
    }
    let len = self.base_pair_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        let dict_min_base_pair = get_dict_min_base_pair(&(i, j));
        if (i, j) != dict_min_base_pair {
          continue;
        }
        sub_len += 1;
      }
    }
    let len = self.interior_loop_length_count_mat_explicit.len();
    for i in 0..len {
      for j in 0..len {
        let dict_min_loop_len_pair = get_dict_min_loop_len_pair(&(i, j));
        if (i, j) != dict_min_loop_len_pair {
          continue;
        }
        sub_len += 1;
      }
    }
    sub_len += self.bulge_loop_0x1_length_counts.len();
    let len = self.interior_loop_1x1_length_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        let dict_min_nuc_pair = get_dict_min_nuc_pair(&(i, j));
        if (i, j) != dict_min_nuc_pair {
          continue;
        }
        sub_len += 1;
      }
    }
    sub_len += CONSPROB_MULTI_LOOP_GROUP_SIZE;
    sub_len += CONSPROB_EXTERNAL_LOOP_GROUP_SIZE;
    sub_len += CONSPROB_MATCH_TRANSITION_GROUP_SIZE;
    sub_len += CONSPROB_INSERT_TRANSITION_GROUP_SIZE;
    sub_len += self.insert_counts.len();
    let len = self.align_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        let dict_min_align = get_dict_min_align(&(i, j));
        if (i, j) != dict_min_align {
          continue;
        }
        sub_len += 1;
      }
    }
    sub_len
  }

  pub fn update_regularizers(&self, regularizers: &mut Regularizers) {
    let mut regularizers_tmp = vec![0.; regularizers.len()];
    let mut offset = 0;
    let len = self.hairpin_loop_length_counts.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0..len {
      let count = self.hairpin_loop_length_counts[i];
      squared_sum += count * count;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0..len {
      regularizers_tmp[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.bulge_loop_length_counts.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0..len {
      let count = self.bulge_loop_length_counts[i];
      squared_sum += count * count;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0..len {
      regularizers_tmp[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.interior_loop_length_counts.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0..len {
      let count = self.interior_loop_length_counts[i];
      squared_sum += count * count;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0..len {
      regularizers_tmp[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.interior_loop_length_counts_symm.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0..len {
      let count = self.interior_loop_length_counts_symm[i];
      squared_sum += count * count;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0..len {
      regularizers_tmp[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.interior_loop_length_counts_asymm.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0..len {
      let count = self.interior_loop_length_counts_asymm[i];
      squared_sum += count * count;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0..len {
      regularizers_tmp[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.stack_count_mat.len();
    let mut effective_group_size = 0;
    let mut squared_sum = 0.;
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for k in 0..len {
          for l in 0..len {
            if !is_canonical(&(k, l)) {
              continue;
            }
            let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
            if ((i, j), (k, l)) != dict_min_stack {
              continue;
            }
            let count = self.stack_count_mat[i][j][k][l];
            squared_sum += count * count;
            effective_group_size += 1;
          }
        }
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for k in 0..len {
          for l in 0..len {
            if !is_canonical(&(k, l)) {
              continue;
            }
            let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
            if ((i, j), (k, l)) != dict_min_stack {
              continue;
            }
            regularizers_tmp[offset] = regularizer;
            offset += 1;
          }
        }
      }
    }
    let len = self.terminal_mismatch_count_mat.len();
    let mut effective_group_size = 0;
    let mut squared_sum = 0.;
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for k in 0..len {
          for l in 0..len {
            let count = self.terminal_mismatch_count_mat[i][j][k][l];
            squared_sum += count * count;
            effective_group_size += 1;
          }
        }
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for _ in 0..len {
          for _ in 0..len {
            regularizers_tmp[offset] = regularizer;
            offset += 1;
          }
        }
      }
    }
    let len = self.left_dangle_count_mat.len();
    let mut effective_group_size = 0;
    let mut squared_sum = 0.;
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for k in 0..len {
          let count = self.left_dangle_count_mat[i][j][k];
          squared_sum += count * count;
          effective_group_size += 1;
        }
      }
    }
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for k in 0..len {
          let count = self.right_dangle_count_mat[i][j][k];
          squared_sum += count * count;
          effective_group_size += 1;
        }
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for _ in 0..len {
          regularizers_tmp[offset] = regularizer;
          offset += 1;
        }
      }
    }
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for _ in 0..len {
          regularizers_tmp[offset] = regularizer;
          offset += 1;
        }
      }
    }
    let len = self.helix_end_count_mat.len();
    let mut effective_group_size = 0;
    let mut squared_sum = 0.;
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        let count = self.helix_end_count_mat[i][j];
        squared_sum += count * count;
        effective_group_size += 1;
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        regularizers_tmp[offset] = regularizer;
        offset += 1;
      }
    }
    let len = self.base_pair_count_mat.len();
    let mut effective_group_size = 0;
    let mut squared_sum = 0.;
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        let dict_min_base_pair = get_dict_min_base_pair(&(i, j));
        if (i, j) != dict_min_base_pair {
          continue;
        }
        let count = self.base_pair_count_mat[i][j];
        squared_sum += count * count;
        effective_group_size += 1;
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        let dict_min_base_pair = get_dict_min_base_pair(&(i, j));
        if (i, j) != dict_min_base_pair {
          continue;
        }
        regularizers_tmp[offset] = regularizer;
        offset += 1;
      }
    }
    let len = self.interior_loop_length_count_mat_explicit.len();
    let mut effective_group_size = 0;
    let mut squared_sum = 0.;
    for i in 0..len {
      for j in 0..len {
        let dict_min_loop_len_pair = get_dict_min_loop_len_pair(&(i, j));
        if (i, j) != dict_min_loop_len_pair {
          continue;
        }
        let count = self.interior_loop_length_count_mat_explicit[i][j];
        squared_sum += count * count;
        effective_group_size += 1;
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0..len {
      for j in 0..len {
        let dict_min_loop_len_pair = get_dict_min_loop_len_pair(&(i, j));
        if (i, j) != dict_min_loop_len_pair {
          continue;
        }
        regularizers_tmp[offset] = regularizer;
        offset += 1;
      }
    }
    let len = self.bulge_loop_0x1_length_counts.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0..len {
      let count = self.bulge_loop_0x1_length_counts[i];
      squared_sum += count * count;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0..len {
      regularizers_tmp[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.interior_loop_1x1_length_count_mat.len();
    let mut effective_group_size = 0;
    let mut squared_sum = 0.;
    for i in 0..len {
      for j in 0..len {
        let dict_min_nuc_pair = get_dict_min_nuc_pair(&(i, j));
        if (i, j) != dict_min_nuc_pair {
          continue;
        }
        let count = self.interior_loop_1x1_length_count_mat[i][j];
        squared_sum += count * count;
        effective_group_size += 1;
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0..len {
      for j in 0..len {
        let dict_min_nuc_pair = get_dict_min_nuc_pair(&(i, j));
        if (i, j) != dict_min_nuc_pair {
          continue;
        }
        regularizers_tmp[offset] = regularizer;
        offset += 1;
      }
    }
    let mut squared_sum = 0.;
    squared_sum += self.multi_loop_base_count * self.multi_loop_base_count;
    squared_sum += self.multi_loop_basepairing_count * self.multi_loop_basepairing_count;
    squared_sum += self.multi_loop_accessible_baseunpairing_count
      * self.multi_loop_accessible_baseunpairing_count;
    let regularizer = get_regularizer(CONSPROB_MULTI_LOOP_GROUP_SIZE, squared_sum);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let mut squared_sum = 0.;
    squared_sum += self.external_loop_accessible_basepairing_count
      * self.external_loop_accessible_basepairing_count;
    squared_sum += self.external_loop_accessible_baseunpairing_count
      * self.external_loop_accessible_baseunpairing_count;
    let regularizer = get_regularizer(CONSPROB_EXTERNAL_LOOP_GROUP_SIZE, squared_sum);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let mut squared_sum = 0.;
    squared_sum += self.match_2_match_count * self.match_2_match_count;
    squared_sum += self.match_2_insert_count * self.match_2_insert_count;
    squared_sum += self.init_match_count * self.init_match_count;
    let regularizer = get_regularizer(CONSPROB_MATCH_TRANSITION_GROUP_SIZE, squared_sum);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let mut squared_sum = 0.;
    squared_sum += self.insert_extend_count * self.insert_extend_count;
    squared_sum += self.init_insert_count * self.init_insert_count;
    let regularizer = get_regularizer(CONSPROB_INSERT_TRANSITION_GROUP_SIZE, squared_sum);
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    regularizers_tmp[offset] = regularizer;
    offset += 1;
    let len = self.insert_counts.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0..len {
      let count = self.insert_counts[i];
      squared_sum += count * count;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0..len {
      regularizers_tmp[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.align_count_mat.len();
    let mut effective_group_size = 0;
    let mut squared_sum = 0.;
    for i in 0..len {
      for j in 0..len {
        let dict_min_align = get_dict_min_align(&(i, j));
        if (i, j) != dict_min_align {
          continue;
        }
        let count = self.align_count_mat[i][j];
        squared_sum += count * count;
        effective_group_size += 1;
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0..len {
      for j in 0..len {
        let dict_min_align = get_dict_min_align(&(i, j));
        if (i, j) != dict_min_align {
          continue;
        }
        regularizers_tmp[offset] = regularizer;
        offset += 1;
      }
    }
    assert!(offset == self.get_len());
    *regularizers = Array1::from(regularizers_tmp);
  }

  pub fn update<T: HashIndex>(
    &mut self,
    train_data: &[TrainDatum<T>],
    regularizers: &mut Regularizers,
  ) {
    let f =
      |_: &BfgsFeatureCounts| self.get_cost(&train_data[..], regularizers) as BfgsFeatureCount;
    let g = |_: &BfgsFeatureCounts| {
      convert_feature_counts_2_bfgs_feature_counts(&self.get_grad(train_data, regularizers))
    };
    match bfgs(
      convert_feature_counts_2_bfgs_feature_counts(&convert_struct_2_vec(self, false)),
      f,
      g,
    ) {
      Ok(solution) => {
        *self = convert_vec_2_struct(
          &convert_bfgs_feature_counts_2_feature_counts(&solution),
          false,
        );
      }
      Err(_) => {
        println!("BFGS failed");
      }
    };
    self.update_regularizers(regularizers);
    self.mirror();
    self.accumulate();
  }

  pub fn mirror(&mut self) {
    for i in 0..NUM_OF_BASES {
      for j in 0..NUM_OF_BASES {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for k in 0..NUM_OF_BASES {
          for l in 0..NUM_OF_BASES {
            if !is_canonical(&(k, l)) {
              continue;
            }
            let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
            if ((i, j), (k, l)) == dict_min_stack {
              continue;
            }
            self.stack_count_mat[i][j][k][l] = self.stack_count_mat[dict_min_stack.0 .0]
              [dict_min_stack.0 .1][dict_min_stack.1 .0][dict_min_stack.1 .1];
          }
        }
      }
    }
    for i in 0..NUM_OF_BASES {
      for j in 0..NUM_OF_BASES {
        if !is_canonical(&(i, j)) {
          continue;
        }
        let dict_min_base_pair = get_dict_min_base_pair(&(i, j));
        if (i, j) == dict_min_base_pair {
          continue;
        }
        self.base_pair_count_mat[i][j] =
          self.base_pair_count_mat[dict_min_base_pair.0][dict_min_base_pair.1];
      }
    }
    let len = self.interior_loop_length_count_mat_explicit.len();
    for i in 0..len {
      for j in 0..len {
        let dict_min_loop_len_pair = get_dict_min_loop_len_pair(&(i, j));
        if (i, j) == dict_min_loop_len_pair {
          continue;
        }
        self.interior_loop_length_count_mat_explicit[i][j] = self
          .interior_loop_length_count_mat_explicit[dict_min_loop_len_pair.0]
          [dict_min_loop_len_pair.1];
      }
    }
    for i in 0..NUM_OF_BASES {
      for j in 0..NUM_OF_BASES {
        let dict_min_nuc_pair = get_dict_min_nuc_pair(&(i, j));
        if (i, j) == dict_min_nuc_pair {
          continue;
        }
        self.interior_loop_1x1_length_count_mat[i][j] =
          self.interior_loop_1x1_length_count_mat[dict_min_nuc_pair.0][dict_min_nuc_pair.1];
      }
    }
    for i in 0..NUM_OF_BASES {
      for j in 0..NUM_OF_BASES {
        let dict_min_align = get_dict_min_align(&(i, j));
        if (i, j) == dict_min_align {
          continue;
        }
        self.align_count_mat[i][j] = self.align_count_mat[dict_min_align.0][dict_min_align.1];
      }
    }
  }

  pub fn accumulate(&mut self) {
    let mut sum = 0.;
    for i in 0..self.hairpin_loop_length_counts_cumulative.len() {
      sum += self.hairpin_loop_length_counts[i];
      self.hairpin_loop_length_counts_cumulative[i] = sum;
    }
    let mut sum = 0.;
    for i in 0..self.bulge_loop_length_counts_cumulative.len() {
      sum += self.bulge_loop_length_counts[i];
      self.bulge_loop_length_counts_cumulative[i] = sum;
    }
    let mut sum = 0.;
    for i in 0..self.interior_loop_length_counts_cumulative.len() {
      sum += self.interior_loop_length_counts[i];
      self.interior_loop_length_counts_cumulative[i] = sum;
    }
    let mut sum = 0.;
    for i in 0..self.interior_loop_length_counts_symm_cumulative.len() {
      sum += self.interior_loop_length_counts_symm[i];
      self.interior_loop_length_counts_symm_cumulative[i] = sum;
    }
    let mut sum = 0.;
    for i in 0..self.interior_loop_length_counts_asymm_cumulative.len() {
      sum += self.interior_loop_length_counts_asymm[i];
      self.interior_loop_length_counts_asymm_cumulative[i] = sum;
    }
  }

  pub fn get_grad<T: HashIndex>(
    &self,
    train_data: &[TrainDatum<T>],
    regularizers: &Regularizers,
  ) -> FeatureCounts {
    let feature_scores = convert_struct_2_vec(self, false);
    let mut grad = FeatureCountSets::new(0.);
    for train_datum in train_data {
      let ref obs = train_datum.observed_feature_count_sets;
      let ref expect = train_datum.expected_feature_count_sets;
      let mut sum = 0.;
      let len = obs.hairpin_loop_length_counts.len();
      for i in (0..len).rev() {
        let obs_count = obs.hairpin_loop_length_counts[i];
        let expect_count = expect.hairpin_loop_length_counts[i];
        sum -= obs_count - expect_count;
        grad.hairpin_loop_length_counts[i] += sum;
      }
      let len = obs.bulge_loop_length_counts.len();
      let mut sum = 0.;
      for i in (0..len).rev() {
        let obs_count = obs.bulge_loop_length_counts[i];
        let expect_count = expect.bulge_loop_length_counts[i];
        sum -= obs_count - expect_count;
        grad.bulge_loop_length_counts[i] += sum;
      }
      let len = obs.interior_loop_length_counts.len();
      let mut sum = 0.;
      for i in (0..len).rev() {
        let obs_count = obs.interior_loop_length_counts[i];
        let expect_count = expect.interior_loop_length_counts[i];
        sum -= obs_count - expect_count;
        grad.interior_loop_length_counts[i] += sum;
      }
      let len = obs.interior_loop_length_counts_symm.len();
      let mut sum = 0.;
      for i in (0..len).rev() {
        let obs_count = obs.interior_loop_length_counts_symm[i];
        let expect_count = expect.interior_loop_length_counts_symm[i];
        sum -= obs_count - expect_count;
        grad.interior_loop_length_counts_symm[i] += sum;
      }
      let len = obs.interior_loop_length_counts_asymm.len();
      let mut sum = 0.;
      for i in (0..len).rev() {
        let obs_count = obs.interior_loop_length_counts_asymm[i];
        let expect_count = expect.interior_loop_length_counts_asymm[i];
        sum -= obs_count - expect_count;
        grad.interior_loop_length_counts_asymm[i] += sum;
      }
      for i in 0..NUM_OF_BASES {
        for j in 0..NUM_OF_BASES {
          if !is_canonical(&(i, j)) {
            continue;
          }
          for k in 0..NUM_OF_BASES {
            for l in 0..NUM_OF_BASES {
              if !is_canonical(&(k, l)) {
                continue;
              }
              let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
              if ((i, j), (k, l)) != dict_min_stack {
                continue;
              }
              let obs_count = obs.stack_count_mat[i][j][k][l];
              let expect_count = expect.stack_count_mat[i][j][k][l];
              grad.stack_count_mat[i][j][k][l] -= obs_count - expect_count;
            }
          }
        }
      }
      for i in 0..NUM_OF_BASES {
        for j in 0..NUM_OF_BASES {
          if !is_canonical(&(i, j)) {
            continue;
          }
          for k in 0..NUM_OF_BASES {
            for l in 0..NUM_OF_BASES {
              let obs_count = obs.terminal_mismatch_count_mat[i][j][k][l];
              let expect_count = expect.terminal_mismatch_count_mat[i][j][k][l];
              grad.terminal_mismatch_count_mat[i][j][k][l] -= obs_count - expect_count;
            }
          }
        }
      }
      for i in 0..NUM_OF_BASES {
        for j in 0..NUM_OF_BASES {
          if !is_canonical(&(i, j)) {
            continue;
          }
          for k in 0..NUM_OF_BASES {
            let obs_count = obs.left_dangle_count_mat[i][j][k];
            let expect_count = expect.left_dangle_count_mat[i][j][k];
            grad.left_dangle_count_mat[i][j][k] -= obs_count - expect_count;
          }
        }
      }
      for i in 0..NUM_OF_BASES {
        for j in 0..NUM_OF_BASES {
          if !is_canonical(&(i, j)) {
            continue;
          }
          for k in 0..NUM_OF_BASES {
            let obs_count = obs.right_dangle_count_mat[i][j][k];
            let expect_count = expect.right_dangle_count_mat[i][j][k];
            grad.right_dangle_count_mat[i][j][k] -= obs_count - expect_count;
          }
        }
      }
      for i in 0..NUM_OF_BASES {
        for j in 0..NUM_OF_BASES {
          if !is_canonical(&(i, j)) {
            continue;
          }
          let obs_count = obs.helix_end_count_mat[i][j];
          let expect_count = expect.helix_end_count_mat[i][j];
          grad.helix_end_count_mat[i][j] -= obs_count - expect_count;
        }
      }
      for i in 0..NUM_OF_BASES {
        for j in 0..NUM_OF_BASES {
          if !is_canonical(&(i, j)) {
            continue;
          }
          let dict_min_base_pair = get_dict_min_base_pair(&(i, j));
          if (i, j) != dict_min_base_pair {
            continue;
          }
          let obs_count = obs.base_pair_count_mat[i][j];
          let expect_count = expect.base_pair_count_mat[i][j];
          grad.base_pair_count_mat[i][j] -= obs_count - expect_count;
        }
      }
      let len = obs.interior_loop_length_count_mat_explicit.len();
      for i in 0..len {
        for j in 0..len {
          let dict_min_loop_len_pair = get_dict_min_loop_len_pair(&(i, j));
          if (i, j) != dict_min_loop_len_pair {
            continue;
          }
          let obs_count = obs.interior_loop_length_count_mat_explicit[i][j];
          let expect_count = expect.interior_loop_length_count_mat_explicit[i][j];
          grad.interior_loop_length_count_mat_explicit[i][j] -= obs_count - expect_count;
        }
      }
      for i in 0..NUM_OF_BASES {
        let obs_count = obs.bulge_loop_0x1_length_counts[i];
        let expect_count = expect.bulge_loop_0x1_length_counts[i];
        grad.bulge_loop_0x1_length_counts[i] -= obs_count - expect_count;
      }
      for i in 0..NUM_OF_BASES {
        for j in 0..NUM_OF_BASES {
          let dict_min_nuc_pair = get_dict_min_nuc_pair(&(i, j));
          if (i, j) != dict_min_nuc_pair {
            continue;
          }
          let obs_count = obs.interior_loop_1x1_length_count_mat[i][j];
          let expect_count = expect.interior_loop_1x1_length_count_mat[i][j];
          grad.interior_loop_1x1_length_count_mat[i][j] -= obs_count - expect_count;
        }
      }
      let obs_count = obs.multi_loop_base_count;
      let expect_count = expect.multi_loop_base_count;
      grad.multi_loop_base_count -= obs_count - expect_count;
      let obs_count = obs.multi_loop_basepairing_count;
      let expect_count = expect.multi_loop_basepairing_count;
      grad.multi_loop_basepairing_count -= obs_count - expect_count;
      let obs_count = obs.multi_loop_accessible_baseunpairing_count;
      let expect_count = expect.multi_loop_accessible_baseunpairing_count;
      grad.multi_loop_accessible_baseunpairing_count -= obs_count - expect_count;
      let obs_count = obs.external_loop_accessible_basepairing_count;
      let expect_count = expect.external_loop_accessible_basepairing_count;
      grad.external_loop_accessible_basepairing_count -= obs_count - expect_count;
      let obs_count = obs.external_loop_accessible_baseunpairing_count;
      let expect_count = expect.external_loop_accessible_baseunpairing_count;
      grad.external_loop_accessible_baseunpairing_count -= obs_count - expect_count;
      let obs_count = obs.match_2_match_count;
      let expect_count = expect.match_2_match_count;
      grad.match_2_match_count -= obs_count - expect_count;
      let obs_count = obs.match_2_insert_count;
      let expect_count = expect.match_2_insert_count;
      grad.match_2_insert_count -= obs_count - expect_count;
      let obs_count = obs.insert_extend_count;
      let expect_count = expect.insert_extend_count;
      grad.insert_extend_count -= obs_count - expect_count;
      let obs_count = obs.init_match_count;
      let expect_count = expect.init_match_count;
      grad.init_match_count -= obs_count - expect_count;
      let obs_count = obs.init_insert_count;
      let expect_count = expect.init_insert_count;
      grad.init_insert_count -= obs_count - expect_count;
      for i in 0..NUM_OF_BASES {
        let obs_count = obs.insert_counts[i];
        let expect_count = expect.insert_counts[i];
        grad.insert_counts[i] -= obs_count - expect_count;
      }
      for i in 0..NUM_OF_BASES {
        for j in 0..NUM_OF_BASES {
          let dict_min_align = get_dict_min_align(&(i, j));
          if (i, j) != dict_min_align {
            continue;
          }
          let obs_count = obs.align_count_mat[i][j];
          let expect_count = expect.align_count_mat[i][j];
          grad.align_count_mat[i][j] -= obs_count - expect_count;
        }
      }
    }
    convert_struct_2_vec(&grad, false) + regularizers.clone() * feature_scores
  }

  pub fn get_cost<T: HashIndex>(
    &self,
    train_data: &[TrainDatum<T>],
    regularizers: &Regularizers,
  ) -> FeatureCount {
    let mut log_likelihood = 0.;
    let feature_scores_cumulative = convert_struct_2_vec(self, true);
    for train_datum in train_data {
      let ref obs = train_datum.observed_feature_count_sets;
      log_likelihood += feature_scores_cumulative.dot(&convert_struct_2_vec(obs, false));
      log_likelihood -= train_datum.part_func;
    }
    let feature_scores = convert_struct_2_vec(self, false);
    let product = regularizers.clone() * feature_scores.clone();
    -log_likelihood + product.dot(&feature_scores) / 2.
  }

  pub fn rand_init(&mut self) {
    let len = self.get_len();
    let std_deviation = 1. / (len as FeatureCount).sqrt();
    let normal = Normal::new(0., std_deviation).unwrap();
    let mut thread_rng = thread_rng();
    for v in self.hairpin_loop_length_counts.iter_mut() {
      *v = normal.sample(&mut thread_rng);
    }
    for v in self.bulge_loop_length_counts.iter_mut() {
      *v = normal.sample(&mut thread_rng);
    }
    for v in self.interior_loop_length_counts.iter_mut() {
      *v = normal.sample(&mut thread_rng);
    }
    for v in self.interior_loop_length_counts_symm.iter_mut() {
      *v = normal.sample(&mut thread_rng);
    }
    for v in self.interior_loop_length_counts_asymm.iter_mut() {
      *v = normal.sample(&mut thread_rng);
    }
    let len = self.stack_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for k in 0..len {
          for l in 0..len {
            if !is_canonical(&(k, l)) {
              continue;
            }
            let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
            if ((i, j), (k, l)) != dict_min_stack {
              continue;
            }
            let v = normal.sample(&mut thread_rng);
            self.stack_count_mat[i][j][k][l] = v;
          }
        }
      }
    }
    let len = self.terminal_mismatch_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for k in 0..len {
          for l in 0..len {
            let v = normal.sample(&mut thread_rng);
            self.terminal_mismatch_count_mat[i][j][k][l] = v;
          }
        }
      }
    }
    let len = self.left_dangle_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for k in 0..len {
          let v = normal.sample(&mut thread_rng);
          self.left_dangle_count_mat[i][j][k] = v;
        }
      }
    }
    let len = self.right_dangle_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for k in 0..len {
          let v = normal.sample(&mut thread_rng);
          self.right_dangle_count_mat[i][j][k] = v;
        }
      }
    }
    let len = self.helix_end_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        let v = normal.sample(&mut thread_rng);
        self.helix_end_count_mat[i][j] = v;
      }
    }
    let len = self.base_pair_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        let dict_min_base_pair = get_dict_min_base_pair(&(i, j));
        if (i, j) != dict_min_base_pair {
          continue;
        }
        let v = normal.sample(&mut thread_rng);
        self.base_pair_count_mat[i][j] = v;
      }
    }
    let len = self.interior_loop_length_count_mat_explicit.len();
    for i in 0..len {
      for j in 0..len {
        let dict_min_loop_len_pair = get_dict_min_loop_len_pair(&(i, j));
        if (i, j) != dict_min_loop_len_pair {
          continue;
        }
        let v = normal.sample(&mut thread_rng);
        self.interior_loop_length_count_mat_explicit[i][j] = v;
      }
    }
    for v in &mut self.bulge_loop_0x1_length_counts {
      *v = normal.sample(&mut thread_rng);
    }
    let len = self.interior_loop_1x1_length_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        let dict_min_nuc_pair = get_dict_min_nuc_pair(&(i, j));
        if (i, j) != dict_min_nuc_pair {
          continue;
        }
        let v = normal.sample(&mut thread_rng);
        self.interior_loop_1x1_length_count_mat[i][j] = v;
      }
    }
    self.multi_loop_base_count = normal.sample(&mut thread_rng);
    self.multi_loop_basepairing_count = normal.sample(&mut thread_rng);
    self.multi_loop_accessible_baseunpairing_count = normal.sample(&mut thread_rng);
    self.external_loop_accessible_basepairing_count = normal.sample(&mut thread_rng);
    self.external_loop_accessible_baseunpairing_count = normal.sample(&mut thread_rng);
    self.match_2_match_count = normal.sample(&mut thread_rng);
    self.match_2_insert_count = normal.sample(&mut thread_rng);
    self.insert_extend_count = normal.sample(&mut thread_rng);
    self.init_match_count = normal.sample(&mut thread_rng);
    self.init_insert_count = normal.sample(&mut thread_rng);
    let len = self.insert_counts.len();
    for i in 0..len {
      let v = normal.sample(&mut thread_rng);
      self.insert_counts[i] = v;
    }
    let len = self.align_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        let dict_min_align = get_dict_min_align(&(i, j));
        if (i, j) != dict_min_align {
          continue;
        }
        let v = normal.sample(&mut thread_rng);
        self.align_count_mat[i][j] = v;
      }
    }
    self.mirror();
    self.accumulate();
  }

  pub fn transfer(&mut self) {
    for (v, &w) in self
      .hairpin_loop_length_counts
      .iter_mut()
      .zip(CONTRA_HL_LENGTH_FES_AT_LEAST.iter())
    {
      *v = w;
    }
    for (v, &w) in self
      .bulge_loop_length_counts
      .iter_mut()
      .zip(CONTRA_BL_LENGTH_FES_AT_LEAST.iter())
    {
      *v = w;
    }
    for (v, &w) in self
      .interior_loop_length_counts
      .iter_mut()
      .zip(CONTRA_IL_LENGTH_FES_AT_LEAST.iter())
    {
      *v = w;
    }
    for (v, &w) in self
      .interior_loop_length_counts_symm
      .iter_mut()
      .zip(CONTRA_IL_SYMM_LENGTH_FES_AT_LEAST.iter())
    {
      *v = w;
    }
    for (v, &w) in self
      .interior_loop_length_counts_asymm
      .iter_mut()
      .zip(CONTRA_IL_ASYMM_LENGTH_FES_AT_LEAST.iter())
    {
      *v = w;
    }
    let len = self.stack_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for k in 0..len {
          for l in 0..len {
            if !is_canonical(&(k, l)) {
              continue;
            }
            self.stack_count_mat[i][j][k][l] = CONTRA_STACK_FES[i][j][k][l];
          }
        }
      }
    }
    let len = self.terminal_mismatch_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for k in 0..len {
          for l in 0..len {
            self.terminal_mismatch_count_mat[i][j][k][l] = CONTRA_TERMINAL_MISMATCH_FES[i][j][k][l];
          }
        }
      }
    }
    let len = self.left_dangle_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for k in 0..len {
          self.left_dangle_count_mat[i][j][k] = CONTRA_LEFT_DANGLE_FES[i][j][k];
        }
      }
    }
    let len = self.right_dangle_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        for k in 0..len {
          self.right_dangle_count_mat[i][j][k] = CONTRA_RIGHT_DANGLE_FES[i][j][k];
        }
      }
    }
    let len = self.helix_end_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        self.helix_end_count_mat[i][j] = CONTRA_HELIX_CLOSING_FES[i][j];
      }
    }
    let len = self.base_pair_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        if !is_canonical(&(i, j)) {
          continue;
        }
        self.base_pair_count_mat[i][j] = CONTRA_BASE_PAIR_FES[i][j];
      }
    }
    let len = self.interior_loop_length_count_mat_explicit.len();
    for i in 0..len {
      for j in 0..len {
        self.interior_loop_length_count_mat_explicit[i][j] = CONTRA_IL_EXPLICIT_FES[i][j];
      }
    }
    for (v, &w) in self
      .bulge_loop_0x1_length_counts
      .iter_mut()
      .zip(CONTRA_BL_0X1_FES.iter())
    {
      *v = w;
    }
    let len = self.interior_loop_1x1_length_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        self.interior_loop_1x1_length_count_mat[i][j] = CONTRA_IL_1X1_FES[i][j];
      }
    }
    self.multi_loop_base_count = CONTRA_ML_BASE_FE;
    self.multi_loop_basepairing_count = CONTRA_ML_PAIRED_FE;
    self.multi_loop_accessible_baseunpairing_count = CONTRA_ML_UNPAIRED_FE;
    self.external_loop_accessible_basepairing_count = CONTRA_EL_PAIRED_FE;
    self.external_loop_accessible_baseunpairing_count = CONTRA_EL_UNPAIRED_FE;
    self.match_2_match_count = MATCH_2_MATCH_SCORE;
    self.match_2_insert_count = MATCH_2_INSERT_SCORE;
    self.insert_extend_count = INSERT_EXTEND_SCORE;
    self.init_match_count = INIT_MATCH_SCORE;
    self.init_insert_count = INIT_INSERT_SCORE;
    let len = self.insert_counts.len();
    for i in 0..len {
      self.insert_counts[i] = INSERT_SCORES[i];
    }
    let len = self.align_count_mat.len();
    for i in 0..len {
      for j in 0..len {
        self.align_count_mat[i][j] = MATCH_SCORE_MAT[i][j];
      }
    }
    self.accumulate();
  }
}

impl<T: HashIndex> TrainDatum<T> {
  pub fn origin() -> TrainDatum<T> {
    TrainDatum {
      seq_pair: (Seq::new(), Seq::new()),
      seq_pair_with_gaps: (Seq::new(), Seq::new()),
      observed_feature_count_sets: FeatureCountSets::new(0.),
      expected_feature_count_sets: FeatureCountSets::new(NEG_INFINITY),
      bpp_mat_pair: (SparseProbMat::<T>::default(), SparseProbMat::<T>::default()),
      max_bp_span_pair: (T::zero(), T::zero()),
      part_func: NEG_INFINITY,
      forward_pos_pair_mat_set: PosPairMatSet::<T>::default(),
      backward_pos_pair_mat_set: PosPairMatSet::<T>::default(),
      pos_quadruple_mat_with_len_pairs: PosQuadrupleMatWithLenPairs::<T>::default(),
      matchable_pos_sets_1: SparsePosSets::<T>::default(),
      matchable_pos_sets_2: SparsePosSets::<T>::default(),
      bp_score_param_set_pair: (BpScoreParamSets::<T>::new(), BpScoreParamSets::<T>::new()),
      align_prob_mat: SparseProbMat::<T>::default(),
      align: PairAlign::<T>::new(),
      acc: NEG_INFINITY,
    }
  }

  pub fn new(
    input_file_path: &Path,
    min_bpp: Prob,
    min_align_prob: Prob,
    align_feature_score_sets: &AlignFeatureCountSets,
  ) -> TrainDatum<T> {
    let fasta_file_reader = Reader::from_file(Path::new(input_file_path)).unwrap();
    let fasta_records: Vec<Record> = fasta_file_reader
      .records()
      .map(|rec| rec.unwrap())
      .collect();
    let cons_second_struct = fasta_records[2].seq();
    let seq_pair = (
      convert_with_gaps(&fasta_records[0].seq()),
      convert_with_gaps(&fasta_records[1].seq()),
    );
    let mut seq_pair_without_gaps = (remove_gaps(&seq_pair.0), remove_gaps(&seq_pair.1));
    let bpp_mat_pair = (
      sparsify_bpp_mat::<T>(
        &mccaskill_algo(
          &seq_pair_without_gaps.0[..],
          false,
          true,
          &StructFeatureCountSets::new(0.),
        )
        .0,
        min_bpp,
      ),
      sparsify_bpp_mat::<T>(
        &mccaskill_algo(
          &seq_pair_without_gaps.1[..],
          false,
          true,
          &StructFeatureCountSets::new(0.),
        )
        .0,
        min_bpp,
      ),
    );
    seq_pair_without_gaps.0.insert(0, PSEUDO_BASE);
    seq_pair_without_gaps.0.push(PSEUDO_BASE);
    seq_pair_without_gaps.1.insert(0, PSEUDO_BASE);
    seq_pair_without_gaps.1.push(PSEUDO_BASE);
    let seq_len_pair = (
      T::from_usize(seq_pair_without_gaps.0.len()).unwrap(),
      T::from_usize(seq_pair_without_gaps.1.len()).unwrap(),
    );
    let align_prob_mat = sparsify_align_prob_mat(
      &durbin_algo(
        &(&seq_pair_without_gaps.0[..], &seq_pair_without_gaps.1[..]),
        align_feature_score_sets,
      ),
      min_align_prob,
    );
    let (
      forward_pos_pair_mat_set,
      backward_pos_pair_mat_set,
      _,
      pos_quadruple_mat_with_len_pairs,
      matchable_pos_sets_1,
      matchable_pos_sets_2,
    ) = get_sparse_pos_sets(
      &(&bpp_mat_pair.0, &bpp_mat_pair.1),
      &align_prob_mat,
      &seq_len_pair,
    );
    let max_bp_span_pair = (
      get_max_bp_span::<T>(&bpp_mat_pair.0),
      get_max_bp_span::<T>(&bpp_mat_pair.1),
    );
    let mut train_datum = TrainDatum {
      seq_pair: seq_pair_without_gaps,
      seq_pair_with_gaps: seq_pair,
      observed_feature_count_sets: FeatureCountSets::new(0.),
      expected_feature_count_sets: FeatureCountSets::new(NEG_INFINITY),
      bpp_mat_pair: bpp_mat_pair,
      max_bp_span_pair: max_bp_span_pair,
      part_func: NEG_INFINITY,
      forward_pos_pair_mat_set: forward_pos_pair_mat_set,
      backward_pos_pair_mat_set: backward_pos_pair_mat_set,
      pos_quadruple_mat_with_len_pairs: pos_quadruple_mat_with_len_pairs,
      matchable_pos_sets_1: matchable_pos_sets_1,
      matchable_pos_sets_2: matchable_pos_sets_2,
      bp_score_param_set_pair: (BpScoreParamSets::<T>::new(), BpScoreParamSets::<T>::new()),
      align_prob_mat: align_prob_mat,
      align: PairAlign::<T>::new(),
      acc: NEG_INFINITY,
    };
    train_datum.convert(cons_second_struct);
    train_datum
  }

  pub fn convert(&mut self, dot_bracket_notation: TextSlice) {
    let align_len = dot_bracket_notation.len();
    let mut is_inserting = false;
    let mut is_inserting_2 = is_inserting;
    let ref seq_pair = self.seq_pair_with_gaps;
    let mut pos_pair = (T::one(), T::one());
    for i in 0..align_len {
      let char_pair = (seq_pair.0[i], seq_pair.1[i]);
      if dot_bracket_notation[i] != UNPAIRING_BASE {
        let dict_min_align = get_dict_min_align(&char_pair);
        self.observed_feature_count_sets.align_count_mat[dict_min_align.0][dict_min_align.1] += 1.;
        if i == 0 {
          self.observed_feature_count_sets.init_match_count += 1.;
        } else {
          if is_inserting || is_inserting_2 {
            self.observed_feature_count_sets.match_2_insert_count += 1.;
          } else {
            self.observed_feature_count_sets.match_2_match_count += 1.;
          }
        }
        is_inserting = false;
        is_inserting_2 = is_inserting;
        if char_pair.1 == PSEUDO_BASE {
          self.align.inserted_poss.insert(pos_pair.0);
          pos_pair.0 = pos_pair.0 + T::one();
        } else if char_pair.0 == PSEUDO_BASE {
          self.align.deleted_poss.insert(pos_pair.1);
          pos_pair.1 = pos_pair.1 + T::one();
        } else {
          self.align.aligned_pos_pair_mat.insert(pos_pair);
          pos_pair.0 = pos_pair.0 + T::one();
          pos_pair.1 = pos_pair.1 + T::one();
        }
        continue;
      }
      if char_pair.1 == PSEUDO_BASE {
        if i == 0 {
          self.observed_feature_count_sets.init_insert_count += 1.;
        } else {
          if is_inserting {
            self.observed_feature_count_sets.insert_extend_count += 1.;
          } else if is_inserting_2 {
            is_inserting = true;
            is_inserting_2 = false;
          } else {
            self.observed_feature_count_sets.match_2_insert_count += 1.;
            is_inserting = true;
          }
        }
        self.observed_feature_count_sets.insert_counts[char_pair.0] += 1.;
        self.align.inserted_poss.insert(pos_pair.0);
        pos_pair.0 = pos_pair.0 + T::one();
      } else if char_pair.0 == PSEUDO_BASE {
        if i == 0 {
          self.observed_feature_count_sets.init_insert_count += 1.;
          is_inserting_2 = true;
        } else {
          if is_inserting_2 {
            self.observed_feature_count_sets.insert_extend_count += 1.;
          } else if is_inserting {
            is_inserting_2 = true;
            is_inserting = false;
          } else {
            self.observed_feature_count_sets.match_2_insert_count += 1.;
            is_inserting_2 = true;
          }
        }
        self.observed_feature_count_sets.insert_counts[char_pair.1] += 1.;
        self.align.deleted_poss.insert(pos_pair.1);
        pos_pair.1 = pos_pair.1 + T::one();
      } else {
        let dict_min_align = get_dict_min_align(&char_pair);
        self.observed_feature_count_sets.align_count_mat[dict_min_align.0][dict_min_align.1] += 1.;
        if i == 0 {
          self.observed_feature_count_sets.init_match_count += 1.;
        } else {
          if is_inserting || is_inserting_2 {
            self.observed_feature_count_sets.match_2_insert_count += 1.;
          } else {
            self.observed_feature_count_sets.match_2_match_count += 1.;
          }
        }
        is_inserting = false;
        is_inserting_2 = is_inserting;
        self.align.aligned_pos_pair_mat.insert(pos_pair);
        pos_pair.0 = pos_pair.0 + T::one();
        pos_pair.1 = pos_pair.1 + T::one();
      }
    }
    let mut stack = Vec::new();
    let mut cons_second_struct = HashSet::<(usize, usize)>::default();
    for i in 0..align_len {
      let notation_char = dot_bracket_notation[i];
      if notation_char == BASE_PAIRING_LEFT_BASE {
        stack.push(i);
      } else if notation_char == BASE_PAIRING_RIGHT_BASE {
        let pos = stack.pop().unwrap();
        cons_second_struct.insert((pos, i));
        let base_pair_1 = (seq_pair.0[pos], seq_pair.0[i]);
        let base_pair_2 = (seq_pair.1[pos], seq_pair.1[i]);
        if is_canonical(&base_pair_1) {
          let dict_min_base_pair_1 = get_dict_min_base_pair(&base_pair_1);
          self.observed_feature_count_sets.base_pair_count_mat[dict_min_base_pair_1.0]
            [dict_min_base_pair_1.1] += 1.;
        }
        if is_canonical(&base_pair_2) {
          let dict_min_base_pair_2 = get_dict_min_base_pair(&base_pair_2);
          self.observed_feature_count_sets.base_pair_count_mat[dict_min_base_pair_2.0]
            [dict_min_base_pair_2.1] += 1.;
        }
      }
    }
    let mut loop_struct = LoopStruct::default();
    let mut stored_pos_pairs = HashSet::<(usize, usize)>::default();
    for substr_len in 2..align_len + 1 {
      for i in 0..align_len - substr_len + 1 {
        let mut is_pos_pair_found = false;
        let j = i + substr_len - 1;
        if cons_second_struct.contains(&(i, j)) {
          is_pos_pair_found = true;
          cons_second_struct.remove(&(i, j));
          stored_pos_pairs.insert((i, j));
        }
        if is_pos_pair_found {
          let mut pos_pairs_in_loop = Vec::new();
          for stored_pos_pair in stored_pos_pairs.iter() {
            if i < stored_pos_pair.0 && stored_pos_pair.1 < j {
              pos_pairs_in_loop.push(*stored_pos_pair);
            }
          }
          for pos_pair in pos_pairs_in_loop.iter() {
            stored_pos_pairs.remove(pos_pair);
          }
          pos_pairs_in_loop.sort();
          loop_struct.insert((i, j), pos_pairs_in_loop);
        }
      }
    }
    for (pos_pair_closing_loop, pos_pairs_in_loop) in loop_struct.iter() {
      let num_of_basepairings_in_loop = pos_pairs_in_loop.len();
      let base_pair = (
        seq_pair.0[pos_pair_closing_loop.0],
        seq_pair.0[pos_pair_closing_loop.1],
      );
      let base_pair_2 = (
        seq_pair.1[pos_pair_closing_loop.0],
        seq_pair.1[pos_pair_closing_loop.1],
      );
      let mismatch_pair = get_mismatch_pair(&seq_pair.0[..], &pos_pair_closing_loop, true);
      let mismatch_pair_2 = get_mismatch_pair(&seq_pair.1[..], &pos_pair_closing_loop, true);
      if num_of_basepairings_in_loop == 0 {
        let hairpin_loop_length_pair = (
          get_hairpin_loop_length(&seq_pair.0[..], &pos_pair_closing_loop),
          get_hairpin_loop_length(&seq_pair.1[..], &pos_pair_closing_loop),
        );
        if is_canonical(&base_pair) {
          self.observed_feature_count_sets.terminal_mismatch_count_mat[base_pair.0][base_pair.1]
            [mismatch_pair.0][mismatch_pair.1] += 1.;
          self.observed_feature_count_sets.helix_end_count_mat[base_pair.0][base_pair.1] += 1.;
        }
        if is_canonical(&base_pair_2) {
          self.observed_feature_count_sets.terminal_mismatch_count_mat[base_pair_2.0]
            [base_pair_2.1][mismatch_pair_2.0][mismatch_pair_2.1] += 1.;
          self.observed_feature_count_sets.helix_end_count_mat[base_pair_2.0][base_pair_2.1] += 1.;
        }
        if hairpin_loop_length_pair.0 <= CONSPROB_MAX_HAIRPIN_LOOP_LEN {
          self.observed_feature_count_sets.hairpin_loop_length_counts
            [hairpin_loop_length_pair.0] += 1.;
        } else {
          self.observed_feature_count_sets.hairpin_loop_length_counts
            [CONSPROB_MAX_HAIRPIN_LOOP_LEN] += 1.;
        }
        if hairpin_loop_length_pair.1 <= CONSPROB_MAX_HAIRPIN_LOOP_LEN {
          self.observed_feature_count_sets.hairpin_loop_length_counts
            [hairpin_loop_length_pair.1] += 1.;
        } else {
          self.observed_feature_count_sets.hairpin_loop_length_counts
            [CONSPROB_MAX_HAIRPIN_LOOP_LEN] += 1.;
        }
      } else if num_of_basepairings_in_loop == 1 {
        let ref pos_pair_in_loop = pos_pairs_in_loop[0];
        let base_pair_3 = (
          seq_pair.0[pos_pair_in_loop.0],
          seq_pair.0[pos_pair_in_loop.1],
        );
        let base_pair_4 = (
          seq_pair.1[pos_pair_in_loop.0],
          seq_pair.1[pos_pair_in_loop.1],
        );
        let twoloop_length_pair =
          get_2loop_length_pair(&seq_pair.0[..], &pos_pair_closing_loop, &pos_pair_in_loop);
        let sum = twoloop_length_pair.0 + twoloop_length_pair.1;
        let twoloop_length_pair_2 =
          get_2loop_length_pair(&seq_pair.1[..], &pos_pair_closing_loop, &pos_pair_in_loop);
        let sum_2 = twoloop_length_pair_2.0 + twoloop_length_pair_2.1;
        let mismatch_pair_3 = get_mismatch_pair(&seq_pair.0[..], pos_pair_in_loop, false);
        let mismatch_pair_4 = get_mismatch_pair(&seq_pair.1[..], pos_pair_in_loop, false);
        if sum == 0 {
          if is_canonical(&base_pair) && is_canonical(&base_pair_3) {
            let dict_min_stack = get_dict_min_stack(&base_pair, &base_pair_3);
            self.observed_feature_count_sets.stack_count_mat[dict_min_stack.0 .0]
              [dict_min_stack.0 .1][dict_min_stack.1 .0][dict_min_stack.1 .1] += 1.;
          }
        } else {
          if twoloop_length_pair.0 == 0 || twoloop_length_pair.1 == 0 {
            if sum <= CONSPROB_MAX_TWOLOOP_LEN {
              self.observed_feature_count_sets.bulge_loop_length_counts[sum - 1] += 1.;
              if sum == 1 {
                let mismatch = if twoloop_length_pair.0 == 0 {
                  mismatch_pair.1
                } else {
                  mismatch_pair.0
                };
                self
                  .observed_feature_count_sets
                  .bulge_loop_0x1_length_counts[mismatch] += 1.;
              }
            } else {
              self.observed_feature_count_sets.bulge_loop_length_counts
                [CONSPROB_MAX_TWOLOOP_LEN - 1] += 1.;
            }
          } else {
            let diff = get_diff(twoloop_length_pair.0, twoloop_length_pair.1);
            if sum <= CONSPROB_MAX_TWOLOOP_LEN {
              self.observed_feature_count_sets.interior_loop_length_counts[sum - 2] += 1.;
              if diff == 0 {
                self
                  .observed_feature_count_sets
                  .interior_loop_length_counts_symm[twoloop_length_pair.0 - 1] += 1.;
              } else {
                self
                  .observed_feature_count_sets
                  .interior_loop_length_counts_asymm[diff - 1] += 1.;
              }
              if twoloop_length_pair.0 == 1 && twoloop_length_pair.1 == 1 {
                let dict_min_mismatch_pair = get_dict_min_mismatch_pair(&mismatch_pair);
                self
                  .observed_feature_count_sets
                  .interior_loop_1x1_length_count_mat[dict_min_mismatch_pair.0]
                  [dict_min_mismatch_pair.1] += 1.;
              }
              if twoloop_length_pair.0 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT
                && twoloop_length_pair.1 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT
              {
                let dict_min_loop_len_pair = get_dict_min_loop_len_pair(&twoloop_length_pair);
                self
                  .observed_feature_count_sets
                  .interior_loop_length_count_mat_explicit[dict_min_loop_len_pair.0 - 1]
                  [dict_min_loop_len_pair.1 - 1] += 1.;
              }
            } else {
              self.observed_feature_count_sets.interior_loop_length_counts
                [CONSPROB_MAX_TWOLOOP_LEN - 2] += 1.;
              if diff == 0 {
                if twoloop_length_pair.0 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_SYMM {
                  self
                    .observed_feature_count_sets
                    .interior_loop_length_counts_symm[twoloop_length_pair.0 - 1] += 1.;
                } else {
                  self
                    .observed_feature_count_sets
                    .interior_loop_length_counts_symm[CONSPROB_MAX_INTERIOR_LOOP_LEN_SYMM - 1] +=
                    1.;
                }
              } else {
                if diff <= CONSPROB_MAX_INTERIOR_LOOP_LEN_ASYMM {
                  self
                    .observed_feature_count_sets
                    .interior_loop_length_counts_asymm[diff - 1] += 1.;
                } else {
                  self
                    .observed_feature_count_sets
                    .interior_loop_length_counts_asymm[CONSPROB_MAX_INTERIOR_LOOP_LEN_ASYMM - 1] +=
                    1.;
                }
              }
            }
          }
          if is_canonical(&base_pair) {
            self.observed_feature_count_sets.terminal_mismatch_count_mat[base_pair.0]
              [base_pair.1][mismatch_pair.0][mismatch_pair.1] += 1.;
            self.observed_feature_count_sets.helix_end_count_mat[base_pair.0][base_pair.1] += 1.;
          }
          if is_canonical(&base_pair_3) {
            self.observed_feature_count_sets.terminal_mismatch_count_mat[base_pair_3.1]
              [base_pair_3.0][mismatch_pair_3.1][mismatch_pair_3.0] += 1.;
            self.observed_feature_count_sets.helix_end_count_mat[base_pair_3.1][base_pair_3.0] +=
              1.;
          }
        }
        if sum_2 == 0 {
          if is_canonical(&base_pair_2) && is_canonical(&base_pair_4) {
            let dict_min_stack_2 = get_dict_min_stack(&base_pair_2, &base_pair_4);
            self.observed_feature_count_sets.stack_count_mat[dict_min_stack_2.0 .0]
              [dict_min_stack_2.0 .1][dict_min_stack_2.1 .0][dict_min_stack_2.1 .1] += 1.;
          }
        } else {
          if twoloop_length_pair_2.0 == 0 || twoloop_length_pair_2.1 == 0 {
            if sum_2 <= CONSPROB_MAX_TWOLOOP_LEN {
              self.observed_feature_count_sets.bulge_loop_length_counts[sum_2 - 1] += 1.;
              if sum_2 == 1 {
                let mismatch_2 = if twoloop_length_pair_2.0 == 0 {
                  mismatch_pair_2.1
                } else {
                  mismatch_pair_2.0
                };
                self
                  .observed_feature_count_sets
                  .bulge_loop_0x1_length_counts[mismatch_2] += 1.;
              }
            } else {
              self.observed_feature_count_sets.bulge_loop_length_counts
                [CONSPROB_MAX_TWOLOOP_LEN - 1] += 1.;
            }
          } else {
            let diff_2 = get_diff(twoloop_length_pair_2.0, twoloop_length_pair_2.1);
            if sum_2 <= CONSPROB_MAX_TWOLOOP_LEN {
              self.observed_feature_count_sets.interior_loop_length_counts[sum_2 - 2] += 1.;
              if diff_2 == 0 {
                self
                  .observed_feature_count_sets
                  .interior_loop_length_counts_symm[twoloop_length_pair_2.0 - 1] += 1.;
              } else {
                self
                  .observed_feature_count_sets
                  .interior_loop_length_counts_asymm[diff_2 - 1] += 1.;
              }
              if twoloop_length_pair_2.0 == 1 && twoloop_length_pair_2.1 == 1 {
                let dict_min_mismatch_pair_2 = get_dict_min_mismatch_pair(&mismatch_pair_2);
                self
                  .observed_feature_count_sets
                  .interior_loop_1x1_length_count_mat[dict_min_mismatch_pair_2.0]
                  [dict_min_mismatch_pair_2.1] += 1.;
              }
              if twoloop_length_pair_2.0 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT
                && twoloop_length_pair_2.1 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT
              {
                let dict_min_loop_len_pair_2 = get_dict_min_loop_len_pair(&twoloop_length_pair_2);
                self
                  .observed_feature_count_sets
                  .interior_loop_length_count_mat_explicit[dict_min_loop_len_pair_2.0 - 1]
                  [dict_min_loop_len_pair_2.1 - 1] += 1.;
              }
            } else {
              self.observed_feature_count_sets.interior_loop_length_counts
                [CONSPROB_MAX_TWOLOOP_LEN - 2] += 1.;
              if diff_2 == 0 {
                if twoloop_length_pair_2.0 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_SYMM {
                  self
                    .observed_feature_count_sets
                    .interior_loop_length_counts_symm[twoloop_length_pair_2.0 - 1] += 1.;
                } else {
                  self
                    .observed_feature_count_sets
                    .interior_loop_length_counts_symm[CONSPROB_MAX_INTERIOR_LOOP_LEN_SYMM - 1] +=
                    1.;
                }
              } else {
                if diff_2 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_ASYMM {
                  self
                    .observed_feature_count_sets
                    .interior_loop_length_counts_asymm[diff_2 - 1] += 1.;
                } else {
                  self
                    .observed_feature_count_sets
                    .interior_loop_length_counts_asymm[CONSPROB_MAX_INTERIOR_LOOP_LEN_ASYMM - 1] +=
                    1.;
                }
              }
            }
          }
          if is_canonical(&base_pair_2) {
            self.observed_feature_count_sets.terminal_mismatch_count_mat[base_pair_2.0]
              [base_pair_2.1][mismatch_pair_2.0][mismatch_pair_2.1] += 1.;
            self.observed_feature_count_sets.helix_end_count_mat[base_pair_2.0][base_pair_2.1] +=
              1.;
          }
          if is_canonical(&base_pair_4) {
            self.observed_feature_count_sets.terminal_mismatch_count_mat[base_pair_4.1]
              [base_pair_4.0][mismatch_pair_4.1][mismatch_pair_4.0] += 1.;
            self.observed_feature_count_sets.helix_end_count_mat[base_pair_4.1][base_pair_4.0] +=
              1.;
          }
        }
      } else {
        if is_canonical(&base_pair) {
          self.observed_feature_count_sets.left_dangle_count_mat[base_pair.0][base_pair.1]
            [mismatch_pair.0] += 1.;
          self.observed_feature_count_sets.right_dangle_count_mat[base_pair.0][base_pair.1]
            [mismatch_pair.1] += 1.;
          self.observed_feature_count_sets.helix_end_count_mat[base_pair.0][base_pair.1] += 1.;
        }
        if is_canonical(&base_pair_2) {
          self.observed_feature_count_sets.left_dangle_count_mat[base_pair_2.0][base_pair_2.1]
            [mismatch_pair_2.0] += 1.;
          self.observed_feature_count_sets.right_dangle_count_mat[base_pair_2.0][base_pair_2.1]
            [mismatch_pair_2.1] += 1.;
          self.observed_feature_count_sets.helix_end_count_mat[base_pair_2.0][base_pair_2.1] += 1.;
        }
        self.observed_feature_count_sets.multi_loop_base_count += 2.;
        self
          .observed_feature_count_sets
          .multi_loop_basepairing_count += 2.;
        self
          .observed_feature_count_sets
          .multi_loop_basepairing_count += 2. * num_of_basepairings_in_loop as Prob;
        let num_of_baseunpairing_nucs_in_multi_loop = get_num_of_multiloop_baseunpairing_nucs(
          pos_pair_closing_loop,
          pos_pairs_in_loop,
          &seq_pair.0[..],
        );
        self
          .observed_feature_count_sets
          .multi_loop_accessible_baseunpairing_count +=
          num_of_baseunpairing_nucs_in_multi_loop as Prob;
        let num_of_baseunpairing_nucs_in_multi_loop_2 = get_num_of_multiloop_baseunpairing_nucs(
          pos_pair_closing_loop,
          pos_pairs_in_loop,
          &seq_pair.1[..],
        );
        self
          .observed_feature_count_sets
          .multi_loop_accessible_baseunpairing_count +=
          num_of_baseunpairing_nucs_in_multi_loop_2 as Prob;
        for pos_pair_in_loop in pos_pairs_in_loop.iter() {
          let base_pair_3 = (
            seq_pair.0[pos_pair_in_loop.0],
            seq_pair.0[pos_pair_in_loop.1],
          );
          let mismatch_pair_3 = get_mismatch_pair(&seq_pair.0[..], pos_pair_in_loop, false);
          let base_pair_4 = (
            seq_pair.1[pos_pair_in_loop.0],
            seq_pair.1[pos_pair_in_loop.1],
          );
          let mismatch_pair_4 = get_mismatch_pair(&seq_pair.1[..], pos_pair_in_loop, false);
          if is_canonical(&base_pair_3) {
            self.observed_feature_count_sets.left_dangle_count_mat[base_pair_3.1][base_pair_3.0]
              [mismatch_pair_3.1] += 1.;
            self.observed_feature_count_sets.right_dangle_count_mat[base_pair_3.1]
              [base_pair_3.0][mismatch_pair_3.0] += 1.;
            self.observed_feature_count_sets.helix_end_count_mat[base_pair_3.1][base_pair_3.0] +=
              1.;
          }
          if is_canonical(&base_pair_4) {
            self.observed_feature_count_sets.left_dangle_count_mat[base_pair_4.1][base_pair_4.0]
              [mismatch_pair_4.1] += 1.;
            self.observed_feature_count_sets.right_dangle_count_mat[base_pair_4.1]
              [base_pair_4.0][mismatch_pair_4.0] += 1.;
            self.observed_feature_count_sets.helix_end_count_mat[base_pair_4.1][base_pair_4.0] +=
              1.;
          }
        }
      }
    }
    self
      .observed_feature_count_sets
      .external_loop_accessible_basepairing_count += 2. * stored_pos_pairs.len() as Prob;
    let mut stored_pos_pairs_sorted = stored_pos_pairs
      .iter()
      .map(|x| *x)
      .collect::<Vec<(usize, usize)>>();
    stored_pos_pairs_sorted.sort();
    let num_of_baseunpairing_nucs_in_external_loop =
      get_num_of_externalloop_baseunpairing_nucs(&stored_pos_pairs_sorted, &seq_pair.0[..]);
    self
      .observed_feature_count_sets
      .external_loop_accessible_baseunpairing_count +=
      num_of_baseunpairing_nucs_in_external_loop as Prob;
    let num_of_baseunpairing_nucs_in_external_loop_2 =
      get_num_of_externalloop_baseunpairing_nucs(&stored_pos_pairs_sorted, &seq_pair.1[..]);
    self
      .observed_feature_count_sets
      .external_loop_accessible_baseunpairing_count +=
      num_of_baseunpairing_nucs_in_external_loop_2 as Prob;
    for pos_pair_in_loop in stored_pos_pairs.iter() {
      let base_pair = (
        seq_pair.0[pos_pair_in_loop.0],
        seq_pair.0[pos_pair_in_loop.1],
      );
      let base_pair_2 = (
        seq_pair.1[pos_pair_in_loop.0],
        seq_pair.1[pos_pair_in_loop.1],
      );
      let mismatch_pair = get_mismatch_pair(&seq_pair.0[..], &pos_pair_in_loop, false);
      let mismatch_pair_2 = get_mismatch_pair(&seq_pair.1[..], &pos_pair_in_loop, false);
      if is_canonical(&base_pair) {
        if mismatch_pair.1 != PSEUDO_BASE {
          self.observed_feature_count_sets.left_dangle_count_mat[base_pair.1][base_pair.0]
            [mismatch_pair.1] += 1.;
        }
        if mismatch_pair.0 != PSEUDO_BASE {
          self.observed_feature_count_sets.right_dangle_count_mat[base_pair.1][base_pair.0]
            [mismatch_pair.0] += 1.;
        }
        self.observed_feature_count_sets.helix_end_count_mat[base_pair.1][base_pair.0] += 1.;
      }
      if is_canonical(&base_pair_2) {
        if mismatch_pair_2.1 != PSEUDO_BASE {
          self.observed_feature_count_sets.left_dangle_count_mat[base_pair_2.1][base_pair_2.0]
            [mismatch_pair_2.1] += 1.;
        }
        if mismatch_pair_2.0 != PSEUDO_BASE {
          self.observed_feature_count_sets.right_dangle_count_mat[base_pair_2.1][base_pair_2.0]
            [mismatch_pair_2.0] += 1.;
        }
        self.observed_feature_count_sets.helix_end_count_mat[base_pair_2.1][base_pair_2.0] += 1.;
      }
    }
  }

  pub fn set_curr_params(&mut self, feature_score_sets: &FeatureCountSets) {
    self.bp_score_param_set_pair.0 = BpScoreParamSets::<T>::set_curr_params(
      feature_score_sets,
      &self.seq_pair.0,
      &self.bpp_mat_pair.0,
    );
    self.bp_score_param_set_pair.1 = BpScoreParamSets::<T>::set_curr_params(
      feature_score_sets,
      &self.seq_pair.1,
      &self.bpp_mat_pair.1,
    );
  }
}

impl InsertScoreRangeSets {
  pub fn origin() -> InsertScoreRangeSets {
    let insert_scores = Vec::new();
    InsertScoreRangeSets {
      insert_scores_range: insert_scores.clone(),
      insert_scores_range_4_el: insert_scores.clone(),
      insert_scores_range_4_ml: insert_scores.clone(),
      insert_scores_range_2: insert_scores.clone(),
      insert_scores_range_4_el_2: insert_scores.clone(),
      insert_scores_range_4_ml_2: insert_scores,
    }
  }

  pub fn new(seq_pair: &SeqPair, feature_score_sets: &FeatureCountSets) -> InsertScoreRangeSets {
    let seq_len_pair = (seq_pair.0.len(), seq_pair.1.len());
    let mut insert_score_range_sets = InsertScoreRangeSets::origin();
    let mat = vec![
      vec![NEG_INFINITY; seq_len_pair.0.to_usize().unwrap()];
      seq_len_pair.0.to_usize().unwrap()
    ];
    insert_score_range_sets.insert_scores_range = mat.clone();
    insert_score_range_sets.insert_scores_range_4_el = mat.clone();
    insert_score_range_sets.insert_scores_range_4_ml = mat;
    let mat = vec![
      vec![NEG_INFINITY; seq_len_pair.1.to_usize().unwrap()];
      seq_len_pair.1.to_usize().unwrap()
    ];
    insert_score_range_sets.insert_scores_range_2 = mat.clone();
    insert_score_range_sets.insert_scores_range_4_el_2 = mat.clone();
    insert_score_range_sets.insert_scores_range_4_ml_2 = mat.clone();
    for i in 1..seq_len_pair.1 - 1 {
      let base = seq_pair.1[i];
      let mut sum = feature_score_sets.insert_counts[base];
      let mut sum_4_el = sum + feature_score_sets.external_loop_accessible_baseunpairing_count;
      let mut sum_4_ml = sum + feature_score_sets.multi_loop_accessible_baseunpairing_count;
      insert_score_range_sets.insert_scores_range_2[i][i] = sum;
      insert_score_range_sets.insert_scores_range_4_el_2[i][i] = sum_4_el;
      insert_score_range_sets.insert_scores_range_4_ml_2[i][i] = sum_4_ml;
      for j in i + 1..seq_len_pair.1 - 1 {
        let base = seq_pair.1[j];
        let term = feature_score_sets.insert_counts[base] + feature_score_sets.insert_extend_count;
        sum += term;
        sum_4_el += term + feature_score_sets.external_loop_accessible_baseunpairing_count;
        sum_4_ml += term + feature_score_sets.multi_loop_accessible_baseunpairing_count;
        insert_score_range_sets.insert_scores_range_2[i][j] = sum;
        insert_score_range_sets.insert_scores_range_4_el_2[i][j] = sum_4_el;
        insert_score_range_sets.insert_scores_range_4_ml_2[i][j] = sum_4_ml;
      }
    }
    for i in 1..seq_len_pair.0 - 1 {
      let base = seq_pair.0[i];
      let term = feature_score_sets.insert_counts[base];
      let mut sum = term;
      let mut sum_4_el = sum + feature_score_sets.external_loop_accessible_baseunpairing_count;
      let mut sum_4_ml = sum + feature_score_sets.multi_loop_accessible_baseunpairing_count;
      insert_score_range_sets.insert_scores_range[i][i] = sum;
      insert_score_range_sets.insert_scores_range_4_el[i][i] = sum_4_el;
      insert_score_range_sets.insert_scores_range_4_ml[i][i] = sum_4_ml;
      for j in i + 1..seq_len_pair.0 - 1 {
        let base = seq_pair.0[j];
        let term = feature_score_sets.insert_counts[base] + feature_score_sets.insert_extend_count;
        sum += term;
        sum_4_el += term + feature_score_sets.external_loop_accessible_baseunpairing_count;
        sum_4_ml += term + feature_score_sets.multi_loop_accessible_baseunpairing_count;
        insert_score_range_sets.insert_scores_range[i][j] = sum;
        insert_score_range_sets.insert_scores_range_4_el[i][j] = sum_4_el;
        insert_score_range_sets.insert_scores_range_4_ml[i][j] = sum_4_ml;
      }
    }
    insert_score_range_sets
  }
}

impl<T: HashIndex> PairAlign<T> {
  pub fn new() -> PairAlign<T> {
    PairAlign {
      aligned_pos_pair_mat: SparsePosMat::<T>::default(),
      inserted_poss: SparsePoss::<T>::default(),
      deleted_poss: SparsePoss::<T>::default(),
    }
  }
}

pub const DEFAULT_MIN_BPP_TRAIN: Prob = DEFAULT_MIN_BPP;
pub const DEFAULT_MIN_ALIGN_PROB_TRAIN: Prob = DEFAULT_MIN_ALIGN_PROB;
pub const NUM_OF_BASEPAIRINGS: usize = 6;
pub const CONSPROB_MAX_HAIRPIN_LOOP_LEN: usize = 30;
pub const CONSPROB_MAX_TWOLOOP_LEN: usize = CONSPROB_MAX_HAIRPIN_LOOP_LEN;
pub const CONSPROB_MIN_HAIRPIN_LOOP_LEN: usize = 3;
pub const CONSPROB_MIN_HAIRPIN_LOOP_SPAN: usize = CONSPROB_MIN_HAIRPIN_LOOP_LEN + 2;
pub const CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT: usize = 4;
pub const CONSPROB_MAX_INTERIOR_LOOP_LEN_SYMM: usize = CONSPROB_MAX_TWOLOOP_LEN / 2;
pub const CONSPROB_MAX_INTERIOR_LOOP_LEN_ASYMM: usize = CONSPROB_MAX_TWOLOOP_LEN - 2;
pub const CONSPROB_MULTI_LOOP_GROUP_SIZE: usize = 3;
pub const CONSPROB_EXTERNAL_LOOP_GROUP_SIZE: usize = 2;
pub const CONSPROB_MATCH_TRANSITION_GROUP_SIZE: usize = 3;
pub const CONSPROB_INSERT_TRANSITION_GROUP_SIZE: usize = 2;
pub const GAMMA_DIST_ALPHA: FeatureCount = 0.;
pub const GAMMA_DIST_BETA: FeatureCount = 1.;
pub const DEFAULT_LEARNING_TOLERANCE: FeatureCount = 0.000_1;
pub const TRAINED_FEATURE_SCORE_SETS_FILE_PATH: &'static str =
  "../src/trained_feature_score_sets.rs";
pub const TRAINED_FEATURE_SCORE_SETS_FILE_PATH_RANDOM_INIT: &'static str =
  "../src/trained_feature_score_sets_random_init.rs";
pub const README_FILE_NAME: &str = "README.md";
#[derive(Clone, Copy)]
pub enum TrainType {
  TrainedTransfer,
  TrainedRandomInit,
  TransferredOnly,
}
pub const DEFAULT_TRAIN_TYPE: &str = "trained_transfer";

pub fn get_sps_expected<T>(
  seq_pair: &SeqPair,
  align: &PairAlign<T>,
  prob_mats: &StaProbMats<T>,
) -> FeatureCount
where
  T: HashIndex,
{
  let mut align_prob_mat = prob_mats.loop_align_prob_mat.clone();
  for (pos_quadruple, &bpap) in &prob_mats.basepair_align_prob_mat {
    let pos_pair = (pos_quadruple.0, pos_quadruple.2);
    match align_prob_mat.get_mut(&pos_pair) {
      Some(bap) => {
        *bap += bpap;
      }
      None => {
        align_prob_mat.insert(pos_pair, bpap);
      }
    }
    let pos_pair = (pos_quadruple.1, pos_quadruple.3);
    match align_prob_mat.get_mut(&pos_pair) {
      Some(bap) => {
        *bap += bpap;
      }
      None => {
        align_prob_mat.insert(pos_pair, bpap);
      }
    }
  }
  let seq_len_pair = (seq_pair.0.len(), seq_pair.1.len());
  let mut insert_probs_pair = (vec![1.; seq_len_pair.0], vec![1.; seq_len_pair.1]);
  for (pos_pair, &align_prob) in &align_prob_mat {
    let long_pos_pair = (
      pos_pair.0.to_usize().unwrap(),
      pos_pair.1.to_usize().unwrap(),
    );
    insert_probs_pair.0[long_pos_pair.0] -= align_prob;
    insert_probs_pair.1[long_pos_pair.1] -= align_prob;
  }
  let total =
    align.aligned_pos_pair_mat.len() + align.inserted_poss.len() + align.deleted_poss.len();
  let mut pt = align
    .aligned_pos_pair_mat
    .iter()
    .map(|x| match align_prob_mat.get(x) {
      Some(&y) => y,
      None => 0.,
    })
    .sum::<FeatureCount>();
  pt += align
    .inserted_poss
    .iter()
    .map(|&x| insert_probs_pair.0[x.to_usize().unwrap()])
    .sum::<FeatureCount>();
  pt += align
    .deleted_poss
    .iter()
    .map(|&x| insert_probs_pair.1[x.to_usize().unwrap()])
    .sum::<FeatureCount>();
  let sps = pt / total as FeatureCount;
  sps
}

pub fn io_algo_4_prob_mats<T>(
  seq_pair: &SeqPair,
  feature_score_sets: &FeatureCountSets,
  max_bp_span_pair: &PosPair<T>,
  align_prob_mat: &SparseProbMat<T>,
  produce_struct_profs: bool,
  train_score_params: bool,
  expected_feature_count_sets: &mut FeatureCountSets,
  forward_pos_pair_mat_set: &PosPairMatSet<T>,
  backward_pos_pair_mat_set: &PosPairMatSet<T>,
  pos_quadruple_mat_with_len_pairs: &PosQuadrupleMatWithLenPairs<T>,
  bp_score_param_set_pair: &BpScoreParamSetPair<T>,
  produce_align_probs: bool,
  matchable_pos_sets_1: &SparsePosSets<T>,
  matchable_pos_sets_2: &SparsePosSets<T>,
) -> (StaProbMats<T>, Prob)
where
  T: HashIndex,
{
  let insert_score_range_sets = InsertScoreRangeSets::new(seq_pair, feature_score_sets);
  let (sta_part_func_mats, global_part_func) = get_sta_inside_part_func_mats::<T>(
    seq_pair,
    feature_score_sets,
    max_bp_span_pair,
    align_prob_mat,
    train_score_params,
    forward_pos_pair_mat_set,
    backward_pos_pair_mat_set,
    pos_quadruple_mat_with_len_pairs,
    bp_score_param_set_pair,
    &insert_score_range_sets,
    matchable_pos_sets_1,
    matchable_pos_sets_2,
  );
  (
    get_sta_prob_mats::<T>(
      seq_pair,
      feature_score_sets,
      max_bp_span_pair,
      align_prob_mat,
      &sta_part_func_mats,
      produce_struct_profs,
      global_part_func,
      train_score_params,
      expected_feature_count_sets,
      pos_quadruple_mat_with_len_pairs,
      bp_score_param_set_pair,
      produce_align_probs,
      forward_pos_pair_mat_set,
      backward_pos_pair_mat_set,
      &insert_score_range_sets,
      matchable_pos_sets_1,
      matchable_pos_sets_2,
    ),
    global_part_func,
  )
}

pub fn get_sta_inside_part_func_mats<T>(
  seq_pair: &SeqPair,
  feature_score_sets: &FeatureCountSets,
  max_bp_span_pair: &PosPair<T>,
  align_prob_mat: &SparseProbMat<T>,
  train_score_params: bool,
  forward_pos_pair_mat_set: &PosPairMatSet<T>,
  backward_pos_pair_mat_set: &PosPairMatSet<T>,
  pos_quadruple_mat_with_len_pairs: &PosQuadrupleMatWithLenPairs<T>,
  bp_score_param_set_pair: &BpScoreParamSetPair<T>,
  insert_score_range_sets: &InsertScoreRangeSets,
  matchable_pos_sets_1: &SparsePosSets<T>,
  matchable_pos_sets_2: &SparsePosSets<T>,
) -> (StaPartFuncMats<T>, PartFunc)
where
  T: HashIndex,
{
  let seq_len_pair = (
    T::from_usize(seq_pair.0.len()).unwrap(),
    T::from_usize(seq_pair.1.len()).unwrap(),
  );
  let mut sta_part_func_mats = StaPartFuncMats::<T>::new();
  for substr_len_1 in range_inclusive(
    T::from_usize(if train_score_params {
      2
    } else {
      CONSPROB_MIN_HAIRPIN_LOOP_SPAN
    })
    .unwrap(),
    max_bp_span_pair.0,
  ) {
    for substr_len_2 in range_inclusive(
      T::from_usize(if train_score_params {
        2
      } else {
        CONSPROB_MIN_HAIRPIN_LOOP_SPAN
      })
      .unwrap(),
      max_bp_span_pair.1,
    ) {
      match pos_quadruple_mat_with_len_pairs.get(&(substr_len_1, substr_len_2)) {
        Some(pos_pairs) => {
          for &(i, k) in pos_pairs {
            let (j, l) = (i + substr_len_1 - T::one(), k + substr_len_2 - T::one());
            let (long_i, long_j, long_k, long_l) = (
              i.to_usize().unwrap(),
              j.to_usize().unwrap(),
              k.to_usize().unwrap(),
              l.to_usize().unwrap(),
            );
            let base_pair = (seq_pair.0[long_i], seq_pair.0[long_j]);
            let base_pair_2 = (seq_pair.1[long_k], seq_pair.1[long_l]);
            let pos_quadruple = (i, j, k, l);
            let (part_func_on_sa, part_func_4_ml) = get_tmp_part_func_set_mat::<T>(
              &seq_pair,
              feature_score_sets,
              align_prob_mat,
              &pos_quadruple,
              &mut sta_part_func_mats,
              true,
              forward_pos_pair_mat_set,
              insert_score_range_sets,
              matchable_pos_sets_1,
              matchable_pos_sets_2,
            );
            let _ = get_tmp_part_func_set_mat::<T>(
              &seq_pair,
              feature_score_sets,
              align_prob_mat,
              &pos_quadruple,
              &mut sta_part_func_mats,
              false,
              backward_pos_pair_mat_set,
              insert_score_range_sets,
              matchable_pos_sets_1,
              matchable_pos_sets_2,
            );
            let mut sum = NEG_INFINITY;
            let basepair_align_score = feature_score_sets.align_count_mat[base_pair.0]
              [base_pair_2.0]
              + feature_score_sets.align_count_mat[base_pair.1][base_pair_2.1];
            if substr_len_1.to_usize().unwrap() - 2 <= CONSPROB_MAX_HAIRPIN_LOOP_LEN
              && substr_len_2.to_usize().unwrap() - 2 <= CONSPROB_MAX_HAIRPIN_LOOP_LEN
            {
              let hairpin_loop_score = bp_score_param_set_pair.0.hairpin_loop_scores[&(i, j)];
              let hairpin_loop_score_2 = bp_score_param_set_pair.1.hairpin_loop_scores[&(k, l)];
              let score = hairpin_loop_score + hairpin_loop_score_2 + part_func_on_sa;
              logsumexp(&mut sum, score);
            }
            let ref forward_tmp_part_func_set_mat =
              sta_part_func_mats.forward_tmp_part_func_set_mats_with_pos_pairs_decode[&(i, k)];
            let ref backward_tmp_part_func_set_mat =
              sta_part_func_mats.backward_tmp_part_func_set_mats_with_pos_pairs_decode[&(j, l)];
            let min = T::from_usize(if train_score_params {
              2
            } else {
              CONSPROB_MIN_HAIRPIN_LOOP_SPAN
            })
            .unwrap();
            let min_len_pair = (
              if substr_len_1 <= min + T::from_usize(CONSPROB_MAX_TWOLOOP_LEN + 2).unwrap() {
                min
              } else {
                substr_len_1 - T::from_usize(CONSPROB_MAX_TWOLOOP_LEN + 2).unwrap()
              },
              if substr_len_2 <= min + T::from_usize(CONSPROB_MAX_TWOLOOP_LEN + 2).unwrap() {
                min
              } else {
                substr_len_2 - T::from_usize(CONSPROB_MAX_TWOLOOP_LEN + 2).unwrap()
              },
            );
            for substr_len_3 in range(min_len_pair.0, substr_len_1 - T::one()) {
              for substr_len_4 in range(min_len_pair.1, substr_len_2 - T::one()) {
                match pos_quadruple_mat_with_len_pairs.get(&(substr_len_3, substr_len_4)) {
                  Some(pos_pairs_2) => {
                    for &(m, o) in pos_pairs_2 {
                      let (n, p) = (m + substr_len_3 - T::one(), o + substr_len_4 - T::one());
                      if !(i < m && n < j) || !(k < o && p < l) {
                        continue;
                      }
                      if m - i - T::one() + j - n - T::one()
                        > T::from_usize(CONSPROB_MAX_TWOLOOP_LEN).unwrap()
                      {
                        continue;
                      }
                      if o - k - T::one() + l - p - T::one()
                        > T::from_usize(CONSPROB_MAX_TWOLOOP_LEN).unwrap()
                      {
                        continue;
                      }
                      let pos_quadruple_2 = (m, n, o, p);
                      match sta_part_func_mats
                        .part_func_4d_mat_4_bpas
                        .get(&pos_quadruple_2)
                      {
                        Some(&part_func) => {
                          let mut forward_term = NEG_INFINITY;
                          let mut backward_term = forward_term;
                          let pos_pair_2 = (m - T::one(), o - T::one());
                          match forward_tmp_part_func_set_mat.get(&pos_pair_2) {
                            Some(part_funcs) => {
                              let part_func = part_funcs.part_func_on_sa;
                              logsumexp(&mut forward_term, part_func);
                            }
                            None => {}
                          }
                          let pos_pair_2 = (n + T::one(), p + T::one());
                          match backward_tmp_part_func_set_mat.get(&pos_pair_2) {
                            Some(part_funcs) => {
                              let part_func = part_funcs.part_func_on_sa;
                              logsumexp(&mut backward_term, part_func);
                            }
                            None => {}
                          }
                          let twoloop_score =
                            bp_score_param_set_pair.0.twoloop_scores[&(i, j, m, n)];
                          let twoloop_score_2 =
                            bp_score_param_set_pair.1.twoloop_scores[&(k, l, o, p)];
                          let part_func_4_2l = twoloop_score
                            + twoloop_score_2
                            + part_func
                            + forward_term
                            + backward_term;
                          logsumexp(&mut sum, part_func_4_2l);
                        }
                        None => {}
                      }
                    }
                  }
                  None => {}
                }
              }
            }
            let multi_loop_closing_basepairing_score =
              bp_score_param_set_pair.0.multi_loop_closing_bp_scores[&(i, j)];
            let multi_loop_closing_basepairing_score_2 =
              bp_score_param_set_pair.1.multi_loop_closing_bp_scores[&(k, l)];
            let score = multi_loop_closing_basepairing_score
              + multi_loop_closing_basepairing_score_2
              + part_func_4_ml;
            logsumexp(&mut sum, score);
            if sum > NEG_INFINITY {
              let sum = sum + basepair_align_score;
              sta_part_func_mats
                .part_func_4d_mat_4_bpas
                .insert(pos_quadruple, sum);
              let external_loop_accessible_basepairing_score =
                bp_score_param_set_pair.0.external_loop_accessible_bp_scores[&(i, j)];
              let external_loop_accessible_basepairing_score_2 =
                bp_score_param_set_pair.1.external_loop_accessible_bp_scores[&(k, l)];
              sta_part_func_mats
                .part_func_4d_mat_4_bpas_accessible_on_els
                .insert(
                  pos_quadruple,
                  sum
                    + external_loop_accessible_basepairing_score
                    + external_loop_accessible_basepairing_score_2,
                );
              let multi_loop_accessible_basepairing_score =
                bp_score_param_set_pair.0.multi_loop_accessible_bp_scores[&(i, j)];
              let multi_loop_accessible_basepairing_score_2 =
                bp_score_param_set_pair.1.multi_loop_accessible_bp_scores[&(k, l)];
              sta_part_func_mats
                .part_func_4d_mat_4_bpas_accessible_on_mls
                .insert(
                  pos_quadruple,
                  sum
                    + multi_loop_accessible_basepairing_score
                    + multi_loop_accessible_basepairing_score_2,
                );
            }
          }
        }
        None => {}
      }
    }
  }
  let leftmost_pos_pair = (T::zero(), T::zero());
  let rightmost_pos_pair = (seq_len_pair.0 - T::one(), seq_len_pair.1 - T::one());
  sta_part_func_mats
    .forward_part_func_mat_4_external_loop
    .insert(leftmost_pos_pair, 0.);
  sta_part_func_mats
    .backward_part_func_mat_4_external_loop
    .insert(rightmost_pos_pair, 0.);
  for i in range(T::zero(), seq_len_pair.0 - T::one()) {
    let long_i = i.to_usize().unwrap();
    let base = seq_pair.0[long_i];
    for j in range(T::zero(), seq_len_pair.1 - T::one()) {
      let pos_pair = (i, j);
      if pos_pair == (T::zero(), T::zero()) {
        continue;
      }
      let long_j = j.to_usize().unwrap();
      let mut sum = NEG_INFINITY;
      match forward_pos_pair_mat_set.get(&pos_pair) {
        Some(forward_pos_pair_mat) => {
          for &(k, l) in forward_pos_pair_mat {
            let pos_pair_2 = (k - T::one(), l - T::one());
            let pos_quadruple = (k, i, l, j);
            match sta_part_func_mats
              .part_func_4d_mat_4_bpas_accessible_on_els
              .get(&pos_quadruple)
            {
              Some(&part_func) => {
                match sta_part_func_mats
                  .forward_part_func_mat_4_external_loop_decode
                  .get(&pos_pair_2)
                {
                  Some(&part_func_2) => {
                    let score = part_func_2 + part_func;
                    logsumexp(&mut sum, score);
                  }
                  None => {}
                }
              }
              None => {}
            }
          }
        }
        None => {}
      }
      let base_2 = seq_pair.1[long_j];
      if i > T::zero() && j > T::zero() {
        if align_prob_mat.contains_key(&pos_pair) {
          let mut sum_2 = NEG_INFINITY;
          let loop_align_score = feature_score_sets.align_count_mat[base][base_2]
            + 2. * feature_score_sets.external_loop_accessible_baseunpairing_count;
          let pos_pair_2 = (i - T::one(), j - T::one());
          let long_pos_pair_2 = (
            pos_pair_2.0.to_usize().unwrap(),
            pos_pair_2.1.to_usize().unwrap(),
          );
          let is_begin = pos_pair_2 == leftmost_pos_pair;
          match sta_part_func_mats
            .forward_part_func_mat_4_external_loop
            .get(&pos_pair_2)
          {
            Some(&part_func) => {
              let score = part_func
                + if is_begin {
                  feature_score_sets.init_match_count
                } else {
                  feature_score_sets.match_2_match_count
                };
              logsumexp(&mut sum_2, score);
            }
            None => {}
          }
          match matchable_pos_sets_1.get(&pos_pair_2.0) {
            Some(matchable_poss) => {
              for &x in matchable_poss {
                if x >= pos_pair_2.1 {
                  continue;
                }
                let pos_pair_3 = (pos_pair_2.0, x);
                match sta_part_func_mats
                  .forward_part_func_mat_4_external_loop
                  .get(&pos_pair_3)
                {
                  Some(&part_func) => {
                    let long_x = x.to_usize().unwrap();
                    let is_begin = pos_pair_3 == leftmost_pos_pair;
                    let insert_score_range = insert_score_range_sets.insert_scores_range_4_el_2
                      [long_x + 1][long_pos_pair_2.1]
                      + if is_begin {
                        feature_score_sets.init_insert_count
                      } else {
                        feature_score_sets.match_2_insert_count
                      };
                    let score =
                      part_func + insert_score_range + feature_score_sets.match_2_insert_count;
                    logsumexp(&mut sum_2, score);
                  }
                  None => {}
                }
              }
            }
            None => {}
          }
          match matchable_pos_sets_2.get(&pos_pair_2.1) {
            Some(matchable_poss) => {
              for &x in matchable_poss {
                if x >= pos_pair_2.0 {
                  continue;
                }
                let pos_pair_3 = (x, pos_pair_2.1);
                match sta_part_func_mats
                  .forward_part_func_mat_4_external_loop
                  .get(&pos_pair_3)
                {
                  Some(&part_func) => {
                    let long_x = x.to_usize().unwrap();
                    let is_begin = pos_pair_3 == leftmost_pos_pair;
                    let insert_score_range = insert_score_range_sets.insert_scores_range_4_el
                      [long_x + 1][long_pos_pair_2.0]
                      + if is_begin {
                        feature_score_sets.init_insert_count
                      } else {
                        feature_score_sets.match_2_insert_count
                      };
                    let score =
                      part_func + insert_score_range + feature_score_sets.match_2_insert_count;
                    logsumexp(&mut sum_2, score);
                  }
                  None => {}
                }
              }
            }
            None => {}
          }
          if sum_2 > NEG_INFINITY {
            sta_part_func_mats
              .forward_part_func_mat_4_external_loop_decode
              .insert(pos_pair_2, sum_2);
          }
          let term = sum_2 + loop_align_score;
          logsumexp(&mut sum, term);
          if sum > NEG_INFINITY {
            sta_part_func_mats
              .forward_part_func_mat_4_external_loop
              .insert(pos_pair, sum);
          }
        }
      }
    }
  }
  let mut global_part_func = NEG_INFINITY;
  let pos_pair_2 = (
    rightmost_pos_pair.0 - T::one(),
    rightmost_pos_pair.1 - T::one(),
  );
  let long_pos_pair_2 = (
    pos_pair_2.0.to_usize().unwrap(),
    pos_pair_2.1.to_usize().unwrap(),
  );
  match sta_part_func_mats
    .forward_part_func_mat_4_external_loop
    .get(&pos_pair_2)
  {
    Some(&part_func) => {
      logsumexp(&mut global_part_func, part_func);
    }
    None => {}
  }
  match matchable_pos_sets_1.get(&pos_pair_2.0) {
    Some(matchable_poss) => {
      for &x in matchable_poss {
        if x >= pos_pair_2.1 {
          continue;
        }
        match sta_part_func_mats
          .forward_part_func_mat_4_external_loop
          .get(&(pos_pair_2.0, x))
        {
          Some(&part_func) => {
            let long_x = x.to_usize().unwrap();
            let insert_score_range = insert_score_range_sets.insert_scores_range_4_el_2[long_x + 1]
              [long_pos_pair_2.1]
              + feature_score_sets.match_2_insert_count;
            let score = part_func + insert_score_range;
            logsumexp(&mut global_part_func, score);
          }
          None => {}
        }
      }
    }
    None => {}
  }
  match matchable_pos_sets_2.get(&pos_pair_2.1) {
    Some(matchable_poss) => {
      for &x in matchable_poss {
        if x >= pos_pair_2.0 {
          continue;
        }
        match sta_part_func_mats
          .forward_part_func_mat_4_external_loop
          .get(&(x, pos_pair_2.1))
        {
          Some(&part_func) => {
            let long_x = x.to_usize().unwrap();
            let insert_score_range = insert_score_range_sets.insert_scores_range_4_el[long_x + 1]
              [long_pos_pair_2.0]
              + feature_score_sets.match_2_insert_count;
            let score = part_func + insert_score_range;
            logsumexp(&mut global_part_func, score);
          }
          None => {}
        }
      }
    }
    None => {}
  }
  for i in range(T::one(), seq_len_pair.0).rev() {
    let long_i = i.to_usize().unwrap();
    let base = seq_pair.0[long_i];
    for j in range(T::one(), seq_len_pair.1).rev() {
      let pos_pair = (i, j);
      if pos_pair == (seq_len_pair.0 - T::one(), seq_len_pair.1 - T::one()) {
        continue;
      }
      let long_j = j.to_usize().unwrap();
      let mut sum = NEG_INFINITY;
      match backward_pos_pair_mat_set.get(&pos_pair) {
        Some(backward_pos_pair_mat) => {
          for &(k, l) in backward_pos_pair_mat {
            let pos_pair_2 = (k + T::one(), l + T::one());
            let pos_quadruple = (i, k, j, l);
            match sta_part_func_mats
              .part_func_4d_mat_4_bpas_accessible_on_els
              .get(&pos_quadruple)
            {
              Some(&part_func) => {
                match sta_part_func_mats
                  .backward_part_func_mat_4_external_loop_decode
                  .get(&pos_pair_2)
                {
                  Some(&part_func_2) => {
                    let score = part_func_2 + part_func;
                    logsumexp(&mut sum, score);
                  }
                  None => {}
                }
              }
              None => {}
            }
          }
        }
        None => {}
      }
      let base_2 = seq_pair.1[long_j];
      if i < seq_len_pair.0 - T::one() && j < seq_len_pair.1 - T::one() {
        let pos_pair_2 = (i + T::one(), j + T::one());
        let long_pos_pair_2 = (
          pos_pair_2.0.to_usize().unwrap(),
          pos_pair_2.1.to_usize().unwrap(),
        );
        let is_end = pos_pair_2 == rightmost_pos_pair;
        if align_prob_mat.contains_key(&pos_pair) {
          let mut sum_2 = NEG_INFINITY;
          let loop_align_score = feature_score_sets.align_count_mat[base][base_2]
            + 2. * feature_score_sets.external_loop_accessible_baseunpairing_count;
          match sta_part_func_mats
            .backward_part_func_mat_4_external_loop
            .get(&pos_pair_2)
          {
            Some(&part_func) => {
              let score = part_func
                + if is_end {
                  0.
                } else {
                  feature_score_sets.match_2_match_count
                };
              logsumexp(&mut sum_2, score);
            }
            None => {}
          }
          match matchable_pos_sets_1.get(&pos_pair_2.0) {
            Some(matchable_poss) => {
              for &x in matchable_poss {
                if x <= pos_pair_2.1 {
                  continue;
                }
                let pos_pair_3 = (pos_pair_2.0, x);
                match sta_part_func_mats
                  .backward_part_func_mat_4_external_loop
                  .get(&pos_pair_3)
                {
                  Some(&part_func) => {
                    let long_x = x.to_usize().unwrap();
                    let is_end = pos_pair_3 == rightmost_pos_pair;
                    let insert_score_range = insert_score_range_sets.insert_scores_range_4_el_2
                      [long_pos_pair_2.1][long_x - 1]
                      + if is_end {
                        0.
                      } else {
                        feature_score_sets.match_2_insert_count
                      };
                    let score =
                      part_func + insert_score_range + feature_score_sets.match_2_insert_count;
                    logsumexp(&mut sum_2, score);
                  }
                  None => {}
                }
              }
            }
            None => {}
          }
          match matchable_pos_sets_2.get(&pos_pair_2.1) {
            Some(matchable_poss) => {
              for &x in matchable_poss {
                if x <= pos_pair_2.0 {
                  continue;
                }
                let pos_pair_3 = (x, pos_pair_2.1);
                match sta_part_func_mats
                  .backward_part_func_mat_4_external_loop
                  .get(&pos_pair_3)
                {
                  Some(&part_func) => {
                    let long_x = x.to_usize().unwrap();
                    let is_end = pos_pair_3 == rightmost_pos_pair;
                    let insert_score_range = insert_score_range_sets.insert_scores_range_4_el
                      [long_pos_pair_2.0][long_x - 1]
                      + if is_end {
                        0.
                      } else {
                        feature_score_sets.match_2_insert_count
                      };
                    let score =
                      part_func + insert_score_range + feature_score_sets.match_2_insert_count;
                    logsumexp(&mut sum_2, score);
                  }
                  None => {}
                }
              }
            }
            None => {}
          }
          if sum_2 > NEG_INFINITY {
            sta_part_func_mats
              .backward_part_func_mat_4_external_loop_decode
              .insert(pos_pair_2, sum_2);
          }
          let term = sum_2 + loop_align_score;
          logsumexp(&mut sum, term);
          if sum > NEG_INFINITY {
            sta_part_func_mats
              .backward_part_func_mat_4_external_loop
              .insert(pos_pair, sum);
          }
        }
      }
    }
  }
  (sta_part_func_mats, global_part_func)
}

pub fn get_tmp_part_func_set_mat<T>(
  seq_pair: &SeqPair,
  feature_score_sets: &FeatureCountSets,
  align_prob_mat: &SparseProbMat<T>,
  pos_quadruple: &PosQuadruple<T>,
  sta_part_func_mats: &mut StaPartFuncMats<T>,
  is_forward: bool,
  pos_pair_mat_set: &PosPairMatSet<T>,
  insert_score_range_sets: &InsertScoreRangeSets,
  matchable_pos_sets_1: &SparsePosSets<T>,
  matchable_pos_sets_2: &SparsePosSets<T>,
) -> (PartFunc, PartFunc)
where
  T: HashIndex,
{
  let &(i, j, k, l) = pos_quadruple;
  let leftmost_pos_pair = if is_forward {
    (i, k)
  } else {
    (i + T::one(), k + T::one())
  };
  let rightmost_pos_pair = if is_forward {
    (j - T::one(), l - T::one())
  } else {
    (j, l)
  };
  let tmp_part_func_set_mats_with_pos_pairs = if is_forward {
    &mut sta_part_func_mats.forward_tmp_part_func_set_mats_with_pos_pairs
  } else {
    &mut sta_part_func_mats.backward_tmp_part_func_set_mats_with_pos_pairs
  };
  let tmp_part_func_set_mats_with_pos_pairs_decode = if is_forward {
    &mut sta_part_func_mats.forward_tmp_part_func_set_mats_with_pos_pairs_decode
  } else {
    &mut sta_part_func_mats.backward_tmp_part_func_set_mats_with_pos_pairs_decode
  };
  if !tmp_part_func_set_mats_with_pos_pairs.contains_key(&if is_forward {
    leftmost_pos_pair
  } else {
    rightmost_pos_pair
  }) {
    tmp_part_func_set_mats_with_pos_pairs.insert(
      if is_forward {
        leftmost_pos_pair
      } else {
        rightmost_pos_pair
      },
      TmpPartFuncSetMat::<T>::new(),
    );
  }
  if !tmp_part_func_set_mats_with_pos_pairs_decode.contains_key(&if is_forward {
    leftmost_pos_pair
  } else {
    rightmost_pos_pair
  }) {
    tmp_part_func_set_mats_with_pos_pairs_decode.insert(
      if is_forward {
        leftmost_pos_pair
      } else {
        rightmost_pos_pair
      },
      TmpPartFuncSetMat::<T>::new(),
    );
  }
  let ref mut tmp_part_func_set_mat = tmp_part_func_set_mats_with_pos_pairs
    .get_mut(&if is_forward {
      leftmost_pos_pair
    } else {
      rightmost_pos_pair
    })
    .unwrap();
  let ref mut tmp_part_func_set_mat_decode = tmp_part_func_set_mats_with_pos_pairs_decode
    .get_mut(&if is_forward {
      leftmost_pos_pair
    } else {
      rightmost_pos_pair
    })
    .unwrap();
  let iter: Poss<T> = if is_forward {
    range(i, j).collect()
  } else {
    range_inclusive(i + T::one(), j).rev().collect()
  };
  let iter_2: Poss<T> = if is_forward {
    range(k, l).collect()
  } else {
    range_inclusive(k + T::one(), l).rev().collect()
  };
  for &u in iter.iter() {
    let long_u = u.to_usize().unwrap();
    let base = seq_pair.0[long_u];
    for &v in iter_2.iter() {
      let pos_pair = (u, v);
      if tmp_part_func_set_mat.contains_key(&pos_pair) {
        continue;
      }
      let mut tmp_part_funcs = TmpPartFuncs::new();
      if (is_forward && u == i && v == k) || (!is_forward && u == j && v == l) {
        tmp_part_funcs.part_func_on_sa = 0.;
        tmp_part_funcs.part_func_on_sa_4_ml = 0.;
        tmp_part_funcs.part_func_on_mls = 0.;
        tmp_part_func_set_mat.insert(pos_pair, tmp_part_funcs);
        continue;
      }
      let long_v = v.to_usize().unwrap();
      let mut sum_4_ml = NEG_INFINITY;
      let mut sum_4_first_bpas_on_mls = sum_4_ml;
      let mut tmp_sum = sum_4_ml;
      // For alignments.
      match pos_pair_mat_set.get(&pos_pair) {
        Some(pos_pair_mat) => {
          for &(m, n) in pos_pair_mat {
            if is_forward {
              if !(i < m && k < n) {
                continue;
              }
            } else {
              if !(m < j && n < l) {
                continue;
              }
            }
            let pos_pair_2 = if is_forward {
              (m - T::one(), n - T::one())
            } else {
              (m + T::one(), n + T::one())
            };
            let pos_quadruple_2 = if is_forward {
              (m, u, n, v)
            } else {
              (u, m, v, n)
            };
            match sta_part_func_mats
              .part_func_4d_mat_4_bpas_accessible_on_mls
              .get(&pos_quadruple_2)
            {
              Some(&part_func) => match tmp_part_func_set_mat_decode.get(&pos_pair_2) {
                Some(part_funcs) => {
                  let part_func_2 = part_funcs.part_func_4_bpas_on_mls;
                  let score = part_func_2 + part_func;
                  logsumexp(&mut sum_4_ml, score);
                  let part_func_2 = part_funcs.part_func_on_sa_4_ml;
                  let score = part_func_2 + part_func;
                  logsumexp(&mut sum_4_first_bpas_on_mls, score);
                }
                None => {}
              },
              None => {}
            }
          }
        }
        None => {}
      }
      let pos_pair_2 = if is_forward {
        (u - T::one(), v - T::one())
      } else {
        (u + T::one(), v + T::one())
      };
      let long_pos_pair_2 = (
        pos_pair_2.0.to_usize().unwrap(),
        pos_pair_2.1.to_usize().unwrap(),
      );
      let base_2 = seq_pair.1[long_v];
      if align_prob_mat.contains_key(&pos_pair) {
        let mut tmp_part_funcs_decode = TmpPartFuncs::new();
        let mut sum_on_sa_decode = NEG_INFINITY;
        let mut sum_on_sa_4_ml_decode = sum_on_sa_decode;
        let mut sum_4_ml_decode = sum_on_sa_decode;
        let mut sum_4_first_bpas_on_mls_decode = sum_on_sa_decode;
        let mut tmp_sum_decode = sum_on_sa_decode;
        let loop_align_score = feature_score_sets.align_count_mat[base][base_2];
        let loop_align_score_ml =
          loop_align_score + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
        match tmp_part_func_set_mat.get(&pos_pair_2) {
          Some(part_funcs) => {
            let part_func = part_funcs.part_func_4_ml;
            let score = part_func + feature_score_sets.match_2_match_count;
            logsumexp(&mut sum_4_ml_decode, score);
            let part_func = part_funcs.part_func_4_first_bpas_on_mls;
            let score = part_func + feature_score_sets.match_2_match_count;
            logsumexp(&mut sum_4_first_bpas_on_mls_decode, score);
            let part_func = part_funcs.part_func_on_sa_4_ml;
            let score = part_func + feature_score_sets.match_2_match_count;
            logsumexp(&mut sum_on_sa_4_ml_decode, score);
            let part_func = part_funcs.part_func_on_sa;
            let score = part_func + feature_score_sets.match_2_match_count;
            logsumexp(&mut sum_on_sa_decode, score);
          }
          None => {}
        }
        match matchable_pos_sets_1.get(&pos_pair_2.0) {
          Some(matchable_poss) => {
            for &x in matchable_poss {
              if is_forward && x >= pos_pair_2.1 || (!is_forward && x <= pos_pair_2.1) {
                continue;
              }
              match tmp_part_func_set_mat.get(&(pos_pair_2.0, x)) {
                Some(part_funcs) => {
                  let long_x = x.to_usize().unwrap();
                  let insert_score_range = if is_forward {
                    insert_score_range_sets.insert_scores_range_2[long_x + 1][long_pos_pair_2.1]
                  } else {
                    insert_score_range_sets.insert_scores_range_2[long_pos_pair_2.1][long_x - 1]
                  } + feature_score_sets.match_2_insert_count;
                  let insert_score_range_4_ml = if is_forward {
                    insert_score_range_sets.insert_scores_range_4_ml_2[long_x + 1]
                      [long_pos_pair_2.1]
                  } else {
                    insert_score_range_sets.insert_scores_range_4_ml_2[long_pos_pair_2.1]
                      [long_x - 1]
                  } + feature_score_sets.match_2_insert_count;
                  let part_func = part_funcs.part_func_4_ml;
                  let score =
                    part_func + feature_score_sets.match_2_insert_count + insert_score_range_4_ml;
                  logsumexp(&mut sum_4_ml_decode, score);
                  let part_func = part_funcs.part_func_4_first_bpas_on_mls;
                  let score =
                    part_func + feature_score_sets.match_2_insert_count + insert_score_range_4_ml;
                  logsumexp(&mut sum_4_first_bpas_on_mls_decode, score);
                  let part_func = part_funcs.part_func_on_sa;
                  let score =
                    part_func + feature_score_sets.match_2_insert_count + insert_score_range;
                  logsumexp(&mut sum_on_sa_decode, score);
                  let part_func = part_funcs.part_func_on_sa_4_ml;
                  let score =
                    part_func + feature_score_sets.match_2_insert_count + insert_score_range_4_ml;
                  logsumexp(&mut sum_on_sa_4_ml_decode, score);
                }
                None => {}
              }
            }
          }
          None => {}
        }
        match matchable_pos_sets_2.get(&pos_pair_2.1) {
          Some(matchable_poss) => {
            for &x in matchable_poss {
              if is_forward && x >= pos_pair_2.0 || (!is_forward && x <= pos_pair_2.0) {
                continue;
              }
              match tmp_part_func_set_mat.get(&(x, pos_pair_2.1)) {
                Some(part_funcs) => {
                  let long_x = x.to_usize().unwrap();
                  let insert_score_range = if is_forward {
                    insert_score_range_sets.insert_scores_range[long_x + 1][long_pos_pair_2.0]
                  } else {
                    insert_score_range_sets.insert_scores_range[long_pos_pair_2.0][long_x - 1]
                  } + feature_score_sets.match_2_insert_count;
                  let insert_score_range_4_ml = if is_forward {
                    insert_score_range_sets.insert_scores_range_4_ml[long_x + 1][long_pos_pair_2.0]
                  } else {
                    insert_score_range_sets.insert_scores_range_4_ml[long_pos_pair_2.0][long_x - 1]
                  } + feature_score_sets.match_2_insert_count;
                  let part_func = part_funcs.part_func_4_ml;
                  let score =
                    part_func + feature_score_sets.match_2_insert_count + insert_score_range_4_ml;
                  logsumexp(&mut sum_4_ml_decode, score);
                  let part_func = part_funcs.part_func_4_first_bpas_on_mls;
                  let score =
                    part_func + feature_score_sets.match_2_insert_count + insert_score_range_4_ml;
                  logsumexp(&mut sum_4_first_bpas_on_mls_decode, score);
                  let part_func = part_funcs.part_func_on_sa;
                  let score =
                    part_func + feature_score_sets.match_2_insert_count + insert_score_range;
                  logsumexp(&mut sum_on_sa_decode, score);
                  let part_func = part_funcs.part_func_on_sa_4_ml;
                  let score =
                    part_func + feature_score_sets.match_2_insert_count + insert_score_range_4_ml;
                  logsumexp(&mut sum_on_sa_4_ml_decode, score);
                }
                None => {}
              }
            }
          }
          None => {}
        }
        tmp_part_funcs_decode.part_func_4_ml = sum_4_ml_decode;
        logsumexp(&mut tmp_sum_decode, sum_4_ml_decode);
        tmp_part_funcs_decode.part_func_4_first_bpas_on_mls = sum_4_first_bpas_on_mls_decode;
        logsumexp(&mut tmp_sum_decode, sum_4_first_bpas_on_mls_decode);
        tmp_part_funcs_decode.part_func_4_bpas_on_mls = tmp_sum_decode;
        tmp_part_funcs_decode.part_func_on_sa = sum_on_sa_decode;
        tmp_part_funcs_decode.part_func_on_sa_4_ml = sum_on_sa_4_ml_decode;
        logsumexp(&mut tmp_sum_decode, sum_on_sa_4_ml_decode);
        tmp_part_funcs_decode.part_func_on_mls = tmp_sum_decode;
        if !is_empty(&tmp_part_funcs_decode) {
          tmp_part_func_set_mat_decode.insert(pos_pair_2, tmp_part_funcs_decode);
        }
        let term = sum_4_ml_decode + loop_align_score_ml;
        logsumexp(&mut sum_4_ml, term);
        tmp_part_funcs.part_func_4_ml = sum_4_ml;
        logsumexp(&mut tmp_sum, sum_4_ml);
        let term = sum_4_first_bpas_on_mls_decode + loop_align_score_ml;
        logsumexp(&mut sum_4_first_bpas_on_mls, term);
        tmp_part_funcs.part_func_4_first_bpas_on_mls = sum_4_first_bpas_on_mls;
        logsumexp(&mut tmp_sum, sum_4_first_bpas_on_mls);
        tmp_part_funcs.part_func_4_bpas_on_mls = tmp_sum;
        let sum_on_sa_4_ml = sum_on_sa_4_ml_decode + loop_align_score_ml;
        tmp_part_funcs.part_func_on_sa_4_ml = sum_on_sa_4_ml;
        logsumexp(&mut tmp_sum, sum_on_sa_4_ml);
        tmp_part_funcs.part_func_on_mls = tmp_sum;
        let sum_on_sa = sum_on_sa_decode + loop_align_score;
        tmp_part_funcs.part_func_on_sa = sum_on_sa;
        if !is_empty(&tmp_part_funcs) {
          tmp_part_func_set_mat.insert(pos_pair, tmp_part_funcs);
        }
      }
    }
  }
  let mut edge_part_func_on_sa = NEG_INFINITY;
  let mut edge_part_func_4_ml = edge_part_func_on_sa;
  if is_forward {
    let pos_pair_2 = rightmost_pos_pair;
    let long_pos_pair_2 = (
      pos_pair_2.0.to_usize().unwrap(),
      pos_pair_2.1.to_usize().unwrap(),
    );
    match tmp_part_func_set_mat.get(&pos_pair_2) {
      Some(part_funcs) => {
        let part_func = part_funcs.part_func_4_ml;
        let score = part_func + feature_score_sets.match_2_match_count;
        logsumexp(&mut edge_part_func_4_ml, score);
        let part_func = part_funcs.part_func_on_sa;
        let score = part_func + feature_score_sets.match_2_match_count;
        logsumexp(&mut edge_part_func_on_sa, score);
      }
      None => {}
    }
    match matchable_pos_sets_1.get(&pos_pair_2.0) {
      Some(matchable_poss) => {
        for &x in matchable_poss {
          if x >= pos_pair_2.1 {
            continue;
          }
          match tmp_part_func_set_mat.get(&(pos_pair_2.0, x)) {
            Some(part_funcs) => {
              let part_func = part_funcs.part_func_4_ml;
              let long_x = x.to_usize().unwrap();
              let insert_score_range = insert_score_range_sets.insert_scores_range_2[long_x + 1]
                [long_pos_pair_2.1]
                + feature_score_sets.match_2_insert_count;
              let insert_score_range_4_ml = insert_score_range_sets.insert_scores_range_4_ml_2
                [long_x + 1][long_pos_pair_2.1]
                + feature_score_sets.match_2_insert_count;
              let score =
                part_func + feature_score_sets.match_2_insert_count + insert_score_range_4_ml;
              logsumexp(&mut edge_part_func_4_ml, score);
              let part_func = part_funcs.part_func_on_sa;
              let score = part_func + feature_score_sets.match_2_insert_count + insert_score_range;
              logsumexp(&mut edge_part_func_on_sa, score);
            }
            None => {}
          }
        }
      }
      None => {}
    }
    match matchable_pos_sets_2.get(&pos_pair_2.1) {
      Some(matchable_poss) => {
        for &x in matchable_poss {
          if x >= pos_pair_2.0 {
            continue;
          }
          match tmp_part_func_set_mat.get(&(x, pos_pair_2.1)) {
            Some(part_funcs) => {
              let part_func = part_funcs.part_func_4_ml;
              let long_x = x.to_usize().unwrap();
              let insert_score_range = insert_score_range_sets.insert_scores_range[long_x + 1]
                [long_pos_pair_2.0]
                + feature_score_sets.match_2_insert_count;
              let insert_score_range_4_ml = insert_score_range_sets.insert_scores_range_4_ml
                [long_x + 1][long_pos_pair_2.0]
                + feature_score_sets.match_2_insert_count;
              let score =
                part_func + feature_score_sets.match_2_insert_count + insert_score_range_4_ml;
              logsumexp(&mut edge_part_func_4_ml, score);
              let part_func = part_funcs.part_func_on_sa;
              let score = part_func + feature_score_sets.match_2_insert_count + insert_score_range;
              logsumexp(&mut edge_part_func_on_sa, score);
            }
            None => {}
          }
        }
      }
      None => {}
    }
    let mut tmp_part_funcs_decode = TmpPartFuncs::new();
    tmp_part_funcs_decode.part_func_4_ml = edge_part_func_4_ml;
    tmp_part_funcs_decode.part_func_on_sa = edge_part_func_on_sa;
    tmp_part_func_set_mat_decode.insert(pos_pair_2, tmp_part_funcs_decode);
  }
  (edge_part_func_on_sa, edge_part_func_4_ml)
}

pub fn get_part_func_mat_2loop<T>(
  seq_pair: &SeqPair,
  feature_score_sets: &FeatureCountSets,
  align_prob_mat: &SparseProbMat<T>,
  pos_quadruple: &PosQuadruple<T>,
  sta_part_func_mats: &StaPartFuncMats<T>,
  is_forward: bool,
  pos_pair_mat_set: &PosPairMatSet<T>,
  bp_score_param_set_pair: &BpScoreParamSetPair<T>,
  insert_score_range_sets: &InsertScoreRangeSets,
  matchable_pos_sets_1: &SparsePosSets<T>,
  matchable_pos_sets_2: &SparsePosSets<T>,
) -> (SparsePartFuncMat<T>, SparsePartFuncMat<T>)
where
  T: HashIndex,
{
  let &(i, j, k, l) = pos_quadruple;
  let leftmost_pos_pair = if is_forward {
    (i, k)
  } else {
    (i + T::one(), k + T::one())
  };
  let rightmost_pos_pair = if is_forward {
    (j - T::one(), l - T::one())
  } else {
    (j, l)
  };
  let tmp_part_func_set_mats_with_pos_pairs_decode = if is_forward {
    &sta_part_func_mats.forward_tmp_part_func_set_mats_with_pos_pairs_decode
  } else {
    &sta_part_func_mats.backward_tmp_part_func_set_mats_with_pos_pairs_decode
  };
  let ref tmp_part_func_set_mat_decode = tmp_part_func_set_mats_with_pos_pairs_decode[&if is_forward
  {
    leftmost_pos_pair
  } else {
    rightmost_pos_pair
  }];
  let iter: Poss<T> = if is_forward {
    range(i, j).collect()
  } else {
    range_inclusive(i + T::one(), j).rev().collect()
  };
  let iter_2: Poss<T> = if is_forward {
    range(k, l).collect()
  } else {
    range_inclusive(k + T::one(), l).rev().collect()
  };
  let mut part_func_mat_4_2loop = SparsePartFuncMat::<T>::default();
  let mut part_func_mat_4_2loop_decode = part_func_mat_4_2loop.clone();
  for &u in iter.iter() {
    let long_u = u.to_usize().unwrap();
    let base = seq_pair.0[long_u];
    for &v in iter_2.iter() {
      let pos_pair = (u, v);
      if (is_forward && u == i && v == k) || (!is_forward && u == j && v == l) {
        continue;
      }
      let long_v = v.to_usize().unwrap();
      let mut sum_4_2loop = NEG_INFINITY;
      // For alignments.
      match pos_pair_mat_set.get(&pos_pair) {
        Some(pos_pair_mat) => {
          for &(m, n) in pos_pair_mat {
            if is_forward {
              if !(i < m && k < n) {
                continue;
              }
            } else {
              if !(m < j && n < l) {
                continue;
              }
            }
            let pos_pair_2 = if is_forward {
              (m - T::one(), n - T::one())
            } else {
              (m + T::one(), n + T::one())
            };
            let pos_quadruple_2 = if is_forward {
              (m, u, n, v)
            } else {
              (u, m, v, n)
            };
            if pos_quadruple_2.0 - i - T::one() + j - pos_quadruple_2.1 - T::one()
              > T::from_usize(CONSPROB_MAX_TWOLOOP_LEN).unwrap()
            {
              continue;
            }
            if pos_quadruple_2.2 - k - T::one() + l - pos_quadruple_2.3 - T::one()
              > T::from_usize(CONSPROB_MAX_TWOLOOP_LEN).unwrap()
            {
              continue;
            }
            match sta_part_func_mats
              .part_func_4d_mat_4_bpas
              .get(&pos_quadruple_2)
            {
              Some(&part_func) => match tmp_part_func_set_mat_decode.get(&pos_pair_2) {
                Some(part_funcs) => {
                  let part_func_2 = part_funcs.part_func_on_sa;
                  let twoloop_score = bp_score_param_set_pair.0.twoloop_scores
                    [&(i, j, pos_quadruple_2.0, pos_quadruple_2.1)];
                  let twoloop_score_2 = bp_score_param_set_pair.1.twoloop_scores
                    [&(k, l, pos_quadruple_2.2, pos_quadruple_2.3)];
                  let score = part_func_2 + part_func + twoloop_score + twoloop_score_2;
                  logsumexp(&mut sum_4_2loop, score);
                }
                None => {}
              },
              None => {}
            }
          }
        }
        None => {}
      }
      let pos_pair_2 = if is_forward {
        (u - T::one(), v - T::one())
      } else {
        (u + T::one(), v + T::one())
      };
      let long_pos_pair_2 = (
        pos_pair_2.0.to_usize().unwrap(),
        pos_pair_2.1.to_usize().unwrap(),
      );
      let base_2 = seq_pair.1[long_v];
      if align_prob_mat.contains_key(&pos_pair) {
        let mut sum_4_2loop_decode = NEG_INFINITY;
        let loop_align_score = feature_score_sets.align_count_mat[base][base_2];
        match part_func_mat_4_2loop.get(&pos_pair_2) {
          Some(&part_func) => {
            let score = part_func + feature_score_sets.match_2_match_count;
            logsumexp(&mut sum_4_2loop_decode, score);
          }
          None => {}
        }
        match matchable_pos_sets_1.get(&pos_pair_2.0) {
          Some(matchable_poss) => {
            for &x in matchable_poss {
              if is_forward && x >= pos_pair_2.1 || (!is_forward && x <= pos_pair_2.1) {
                continue;
              }
              match part_func_mat_4_2loop.get(&(pos_pair_2.0, x)) {
                Some(&part_func) => {
                  let long_x = x.to_usize().unwrap();
                  let insert_score_range = if is_forward {
                    insert_score_range_sets.insert_scores_range_2[long_x + 1][long_pos_pair_2.1]
                  } else {
                    insert_score_range_sets.insert_scores_range_2[long_pos_pair_2.1][long_x - 1]
                  } + feature_score_sets.match_2_insert_count;
                  let score =
                    part_func + feature_score_sets.match_2_insert_count + insert_score_range;
                  logsumexp(&mut sum_4_2loop_decode, score);
                }
                None => {}
              }
            }
          }
          None => {}
        }
        match matchable_pos_sets_2.get(&pos_pair_2.1) {
          Some(matchable_poss) => {
            for &x in matchable_poss {
              if is_forward && x >= pos_pair_2.0 || (!is_forward && x <= pos_pair_2.0) {
                continue;
              }
              match part_func_mat_4_2loop.get(&(x, pos_pair_2.1)) {
                Some(&part_func) => {
                  let long_x = x.to_usize().unwrap();
                  let insert_score_range = if is_forward {
                    insert_score_range_sets.insert_scores_range[long_x + 1][long_pos_pair_2.0]
                  } else {
                    insert_score_range_sets.insert_scores_range[long_pos_pair_2.0][long_x - 1]
                  } + feature_score_sets.match_2_insert_count;
                  let score =
                    part_func + feature_score_sets.match_2_insert_count + insert_score_range;
                  logsumexp(&mut sum_4_2loop_decode, score);
                }
                None => {}
              }
            }
          }
          None => {}
        }
        if sum_4_2loop_decode > NEG_INFINITY {
          part_func_mat_4_2loop_decode.insert(pos_pair_2, sum_4_2loop_decode);
        }
        let term = sum_4_2loop_decode + loop_align_score;
        logsumexp(&mut sum_4_2loop, term);
        if sum_4_2loop > NEG_INFINITY {
          part_func_mat_4_2loop.insert(pos_pair, sum_4_2loop);
        }
      }
    }
  }
  (part_func_mat_4_2loop, part_func_mat_4_2loop_decode)
}

pub fn get_sta_prob_mats<T>(
  seq_pair: &SeqPair,
  feature_score_sets: &FeatureCountSets,
  max_bp_span_pair: &PosPair<T>,
  align_prob_mat: &SparseProbMat<T>,
  sta_part_func_mats: &StaPartFuncMats<T>,
  produce_struct_profs: bool,
  global_part_func: PartFunc,
  train_score_params: bool,
  expected_feature_count_sets: &mut FeatureCountSets,
  pos_quadruple_mat_with_len_pairs: &PosQuadrupleMatWithLenPairs<T>,
  bp_score_param_set_pair: &BpScoreParamSetPair<T>,
  produce_align_probs: bool,
  forward_pos_pair_mat_set: &PosPairMatSet<T>,
  backward_pos_pair_mat_set: &PosPairMatSet<T>,
  insert_score_range_sets: &InsertScoreRangeSets,
  matchable_pos_sets_1: &SparsePosSets<T>,
  matchable_pos_sets_2: &SparsePosSets<T>,
) -> StaProbMats<T>
where
  T: HashIndex,
{
  let seq_len_pair = (
    T::from_usize(seq_pair.0.len()).unwrap(),
    T::from_usize(seq_pair.1.len()).unwrap(),
  );
  let mut sta_outside_part_func_4d_mat_4_bpas = PartFunc4dMat::<T>::default();
  let mut sta_prob_mats = StaProbMats::<T>::new(&seq_len_pair);
  let leftmost_pos_pair = (T::zero(), T::zero());
  let rightmost_pos_pair = (seq_len_pair.0 - T::one(), seq_len_pair.1 - T::one());
  let mut prob_coeff_mat_4_ml = PartFunc4dMat::<T>::default();
  let mut prob_coeff_mat_4_ml_2 = prob_coeff_mat_4_ml.clone();
  for substr_len_1 in range_inclusive(
    T::from_usize(if train_score_params {
      2
    } else {
      CONSPROB_MIN_HAIRPIN_LOOP_SPAN
    })
    .unwrap(),
    max_bp_span_pair.0,
  )
  .rev()
  {
    for substr_len_2 in range_inclusive(
      T::from_usize(if train_score_params {
        2
      } else {
        CONSPROB_MIN_HAIRPIN_LOOP_SPAN
      })
      .unwrap(),
      max_bp_span_pair.1,
    )
    .rev()
    {
      match pos_quadruple_mat_with_len_pairs.get(&(substr_len_1, substr_len_2)) {
        Some(pos_pairs) => {
          for &(i, k) in pos_pairs {
            let (j, l) = (i + substr_len_1 - T::one(), k + substr_len_2 - T::one());
            let pos_quadruple = (i, j, k, l);
            match sta_part_func_mats
              .part_func_4d_mat_4_bpas
              .get(&pos_quadruple)
            {
              Some(&part_func_4_bpa) => {
                let (long_i, long_j, long_k, long_l) = (
                  i.to_usize().unwrap(),
                  j.to_usize().unwrap(),
                  k.to_usize().unwrap(),
                  l.to_usize().unwrap(),
                );
                let base_pair = (seq_pair.0[long_i], seq_pair.0[long_j]);
                let base_pair_2 = (seq_pair.1[long_k], seq_pair.1[long_l]);
                let mismatch_pair = (seq_pair.0[long_i - 1], seq_pair.0[long_j + 1]);
                let mismatch_pair_2 = (seq_pair.1[long_k - 1], seq_pair.1[long_l + 1]);
                let prob_coeff = part_func_4_bpa - global_part_func;
                let mut sum = NEG_INFINITY;
                let mut forward_term = sum;
                let mut forward_term_4_align = sum;
                let mut backward_term = sum;
                let pos_pair_2 = (i - T::one(), k - T::one());
                match sta_part_func_mats
                  .forward_part_func_mat_4_external_loop_decode
                  .get(&pos_pair_2)
                {
                  Some(&part_func) => {
                    logsumexp(&mut forward_term, part_func);
                  }
                  None => {}
                }
                if train_score_params {
                  match sta_part_func_mats
                    .forward_part_func_mat_4_external_loop
                    .get(&pos_pair_2)
                  {
                    Some(&part_func) => {
                      let is_begin = pos_pair_2 == leftmost_pos_pair;
                      let term = part_func
                        + if is_begin {
                          feature_score_sets.init_match_count
                        } else {
                          feature_score_sets.match_2_match_count
                        };
                      logsumexp(&mut forward_term_4_align, term);
                    }
                    None => {}
                  }
                }
                let pos_pair_2 = (j + T::one(), l + T::one());
                match sta_part_func_mats
                  .backward_part_func_mat_4_external_loop_decode
                  .get(&pos_pair_2)
                {
                  Some(&part_func) => {
                    logsumexp(&mut backward_term, part_func);
                  }
                  None => {}
                }
                let coefficient = sta_part_func_mats.part_func_4d_mat_4_bpas_accessible_on_els
                  [&pos_quadruple]
                  - part_func_4_bpa;
                if train_score_params {
                  let is_begin = (i - T::one(), k - T::one()) == leftmost_pos_pair;
                  let prob = prob_coeff + coefficient + forward_term_4_align + backward_term;
                  if is_begin {
                    logsumexp(&mut expected_feature_count_sets.init_match_count, prob);
                  } else {
                    logsumexp(&mut expected_feature_count_sets.match_2_match_count, prob);
                  }
                }
                let part_func_4_el = forward_term + backward_term;
                if part_func_4_el > NEG_INFINITY {
                  sum = coefficient + part_func_4_el;
                  let bpap_4_el = prob_coeff + sum;
                  if train_score_params {
                    // Count external loop accessible basepairings.
                    logsumexp(
                      &mut expected_feature_count_sets.external_loop_accessible_basepairing_count,
                      (2. as Prob).ln() + bpap_4_el,
                    );
                    // Count helix ends.
                    logsumexp(
                      &mut expected_feature_count_sets.helix_end_count_mat[base_pair.1]
                        [base_pair.0],
                      bpap_4_el,
                    );
                    logsumexp(
                      &mut expected_feature_count_sets.helix_end_count_mat[base_pair_2.1]
                        [base_pair_2.0],
                      bpap_4_el,
                    );
                    // Count external loop terminal mismatches.
                    if j < seq_len_pair.0 - T::from_usize(2).unwrap() {
                      logsumexp(
                        &mut expected_feature_count_sets.left_dangle_count_mat[base_pair.1]
                          [base_pair.0][mismatch_pair.1],
                        bpap_4_el,
                      );
                    }
                    if i > T::one() {
                      logsumexp(
                        &mut expected_feature_count_sets.right_dangle_count_mat[base_pair.1]
                          [base_pair.0][mismatch_pair.0],
                        bpap_4_el,
                      );
                    }
                    if l < seq_len_pair.1 - T::from_usize(2).unwrap() {
                      logsumexp(
                        &mut expected_feature_count_sets.left_dangle_count_mat[base_pair_2.1]
                          [base_pair_2.0][mismatch_pair_2.1],
                        bpap_4_el,
                      );
                    }
                    if k > T::one() {
                      logsumexp(
                        &mut expected_feature_count_sets.right_dangle_count_mat[base_pair_2.1]
                          [base_pair_2.0][mismatch_pair_2.0],
                        bpap_4_el,
                      );
                    }
                  }
                }
                for substr_len_3 in range_inclusive(
                  substr_len_1 + T::from_usize(2).unwrap(),
                  (substr_len_1 + T::from_usize(CONSPROB_MAX_TWOLOOP_LEN + 2).unwrap())
                    .min(max_bp_span_pair.0),
                ) {
                  for substr_len_4 in range_inclusive(
                    substr_len_2 + T::from_usize(2).unwrap(),
                    (substr_len_2 + T::from_usize(CONSPROB_MAX_TWOLOOP_LEN + 2).unwrap())
                      .min(max_bp_span_pair.1),
                  ) {
                    match pos_quadruple_mat_with_len_pairs.get(&(substr_len_3, substr_len_4)) {
                      Some(pos_pairs_2) => {
                        for &(m, o) in pos_pairs_2 {
                          let (n, p) = (m + substr_len_3 - T::one(), o + substr_len_4 - T::one());
                          if !(m < i && j < n) || !(o < k && l < p) {
                            continue;
                          }
                          let (long_m, long_n, long_o, long_p) = (
                            m.to_usize().unwrap(),
                            n.to_usize().unwrap(),
                            o.to_usize().unwrap(),
                            p.to_usize().unwrap(),
                          );
                          let loop_len_pair = (long_i - long_m - 1, long_n - long_j - 1);
                          let loop_len_pair_2 = (long_k - long_o - 1, long_p - long_l - 1);
                          if loop_len_pair.0 + loop_len_pair.1 > CONSPROB_MAX_TWOLOOP_LEN {
                            continue;
                          }
                          if loop_len_pair_2.0 + loop_len_pair_2.1 > CONSPROB_MAX_TWOLOOP_LEN {
                            continue;
                          }
                          let base_pair_3 = (seq_pair.0[long_m], seq_pair.0[long_n]);
                          let base_pair_4 = (seq_pair.1[long_o], seq_pair.1[long_p]);
                          let is_stack = loop_len_pair.0 == 0 && loop_len_pair.1 == 0;
                          let is_bulge_loop =
                            (loop_len_pair.0 == 0 || loop_len_pair.1 == 0) && !is_stack;
                          let mismatch_pair_3 = (seq_pair.0[long_m + 1], seq_pair.0[long_n - 1]);
                          let pos_quadruple_2 = (m, n, o, p);
                          match sta_outside_part_func_4d_mat_4_bpas.get(&pos_quadruple_2) {
                            Some(&part_func) => {
                              let is_stack_2 = loop_len_pair_2.0 == 0 && loop_len_pair_2.1 == 0;
                              let is_bulge_loop_2 =
                                (loop_len_pair_2.0 == 0 || loop_len_pair_2.1 == 0) && !is_stack_2;
                              let mismatch_pair_4 =
                                (seq_pair.1[long_o + 1], seq_pair.1[long_p - 1]);
                              let ref forward_tmp_part_func_set_mat = sta_part_func_mats
                                .forward_tmp_part_func_set_mats_with_pos_pairs[&(m, o)];
                              let ref forward_tmp_part_func_set_mat_decode = sta_part_func_mats
                                .forward_tmp_part_func_set_mats_with_pos_pairs_decode[&(m, o)];
                              let ref backward_tmp_part_func_set_mat_decode = sta_part_func_mats
                                .backward_tmp_part_func_set_mats_with_pos_pairs_decode[&(n, p)];
                              let mut forward_term = NEG_INFINITY;
                              let mut forward_term_4_align = forward_term;
                              let mut backward_term = forward_term;
                              let pos_pair_2 = (i - T::one(), k - T::one());
                              match forward_tmp_part_func_set_mat_decode.get(&pos_pair_2) {
                                Some(part_funcs) => {
                                  let part_func = part_funcs.part_func_on_sa;
                                  logsumexp(&mut forward_term, part_func);
                                }
                                None => {}
                              }
                              if train_score_params {
                                match forward_tmp_part_func_set_mat.get(&pos_pair_2) {
                                  Some(part_funcs) => {
                                    let part_func = part_funcs.part_func_on_sa;
                                    let term = part_func + feature_score_sets.match_2_match_count;
                                    logsumexp(&mut forward_term_4_align, term);
                                  }
                                  None => {}
                                }
                              }
                              let pos_pair_2 = (j + T::one(), l + T::one());
                              match backward_tmp_part_func_set_mat_decode.get(&pos_pair_2) {
                                Some(part_funcs) => {
                                  let part_func = part_funcs.part_func_on_sa;
                                  logsumexp(&mut backward_term, part_func);
                                }
                                None => {}
                              }
                              let basepair_align_score = feature_score_sets.align_count_mat
                                [base_pair_3.0][base_pair_4.0]
                                + feature_score_sets.align_count_mat[base_pair_3.1][base_pair_4.1];
                              let twoloop_score =
                                bp_score_param_set_pair.0.twoloop_scores[&(m, n, i, j)];
                              let twoloop_score_2 =
                                bp_score_param_set_pair.1.twoloop_scores[&(o, p, k, l)];
                              let coefficient =
                                basepair_align_score + twoloop_score + twoloop_score_2 + part_func;
                              if train_score_params {
                                let prob =
                                  prob_coeff + coefficient + forward_term_4_align + backward_term;
                                logsumexp(
                                  &mut expected_feature_count_sets.match_2_match_count,
                                  prob,
                                );
                              }
                              let part_func_4_2l = forward_term + backward_term;
                              if part_func_4_2l > NEG_INFINITY {
                                let part_func_4_2l = coefficient + part_func_4_2l;
                                logsumexp(&mut sum, part_func_4_2l);
                                let bpap_4_2l = prob_coeff + part_func_4_2l;
                                if produce_struct_profs {
                                  for q in long_m + 1..long_i {
                                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_2l.0[q], bpap_4_2l);
                                  }
                                  for q in long_j + 1..long_n {
                                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_2l.0[q], bpap_4_2l);
                                  }
                                  for q in long_o + 1..long_k {
                                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_2l.1[q], bpap_4_2l);
                                  }
                                  for q in long_l + 1..long_p {
                                    logsumexp(&mut sta_prob_mats.upp_mat_pair_4_2l.1[q], bpap_4_2l);
                                  }
                                }
                                if train_score_params {
                                  if is_stack {
                                    // Count a stack.
                                    let dict_min_stack =
                                      get_dict_min_stack(&base_pair_3, &base_pair);
                                    logsumexp(
                                      &mut expected_feature_count_sets.stack_count_mat
                                        [dict_min_stack.0 .0][dict_min_stack.0 .1]
                                        [dict_min_stack.1 .0][dict_min_stack.1 .1],
                                      bpap_4_2l,
                                    );
                                  } else {
                                    if is_bulge_loop {
                                      // Count a bulge loop length.
                                      let bulge_loop_len = if loop_len_pair.0 == 0 {
                                        loop_len_pair.1
                                      } else {
                                        loop_len_pair.0
                                      };
                                      logsumexp(
                                        &mut expected_feature_count_sets.bulge_loop_length_counts
                                          [bulge_loop_len - 1],
                                        bpap_4_2l,
                                      );
                                      // Count a 0x1 bulge loop.
                                      if bulge_loop_len == 1 {
                                        let mismatch = if loop_len_pair.0 == 0 {
                                          mismatch_pair_3.1
                                        } else {
                                          mismatch_pair_3.0
                                        };
                                        logsumexp(
                                          &mut expected_feature_count_sets
                                            .bulge_loop_0x1_length_counts[mismatch],
                                          bpap_4_2l,
                                        );
                                      }
                                    } else {
                                      // Count an interior loop length.
                                      logsumexp(
                                        &mut expected_feature_count_sets
                                          .interior_loop_length_counts
                                          [loop_len_pair.0 + loop_len_pair.1 - 2],
                                        bpap_4_2l,
                                      );
                                      let diff = get_diff(loop_len_pair.0, loop_len_pair.1);
                                      if diff == 0 {
                                        logsumexp(
                                          &mut expected_feature_count_sets
                                            .interior_loop_length_counts_symm[loop_len_pair.0 - 1],
                                          bpap_4_2l,
                                        );
                                      } else {
                                        logsumexp(
                                          &mut expected_feature_count_sets
                                            .interior_loop_length_counts_asymm[diff - 1],
                                          bpap_4_2l,
                                        );
                                      }
                                      // Count a 1x1 interior loop.
                                      if loop_len_pair.0 == 1 && loop_len_pair.1 == 1 {
                                        let dict_min_mismatch_pair_3 =
                                          get_dict_min_mismatch_pair(&mismatch_pair_3);
                                        logsumexp(
                                          &mut expected_feature_count_sets
                                            .interior_loop_1x1_length_count_mat
                                            [dict_min_mismatch_pair_3.0]
                                            [dict_min_mismatch_pair_3.1],
                                          bpap_4_2l,
                                        );
                                      }
                                      // Count an explicit interior loop length pair.
                                      if loop_len_pair.0 <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT
                                        && loop_len_pair.1
                                          <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT
                                      {
                                        let dict_min_loop_len_pair =
                                          get_dict_min_loop_len_pair(&loop_len_pair);
                                        logsumexp(
                                          &mut expected_feature_count_sets
                                            .interior_loop_length_count_mat_explicit
                                            [dict_min_loop_len_pair.0 - 1]
                                            [dict_min_loop_len_pair.1 - 1],
                                          bpap_4_2l,
                                        );
                                      }
                                    }
                                    // Count helix ends.
                                    logsumexp(
                                      &mut expected_feature_count_sets.helix_end_count_mat
                                        [base_pair.1][base_pair.0],
                                      bpap_4_2l,
                                    );
                                    logsumexp(
                                      &mut expected_feature_count_sets.helix_end_count_mat
                                        [base_pair_3.0][base_pair_3.1],
                                      bpap_4_2l,
                                    );
                                    // Count 2-loop terminal mismatches.
                                    logsumexp(
                                      &mut expected_feature_count_sets.terminal_mismatch_count_mat
                                        [base_pair_3.0][base_pair_3.1][mismatch_pair_3.0]
                                        [mismatch_pair_3.1],
                                      bpap_4_2l,
                                    );
                                    logsumexp(
                                      &mut expected_feature_count_sets.terminal_mismatch_count_mat
                                        [base_pair.1][base_pair.0][mismatch_pair.1]
                                        [mismatch_pair.0],
                                      bpap_4_2l,
                                    );
                                  }
                                  if is_stack_2 {
                                    // Count a stack.
                                    let dict_min_stack_2 =
                                      get_dict_min_stack(&base_pair_4, &base_pair_2);
                                    logsumexp(
                                      &mut expected_feature_count_sets.stack_count_mat
                                        [dict_min_stack_2.0 .0][dict_min_stack_2.0 .1]
                                        [dict_min_stack_2.1 .0][dict_min_stack_2.1 .1],
                                      bpap_4_2l,
                                    );
                                  } else {
                                    if is_bulge_loop_2 {
                                      // Count a bulge loop length.
                                      let bulge_loop_len_2 = if loop_len_pair_2.0 == 0 {
                                        loop_len_pair_2.1
                                      } else {
                                        loop_len_pair_2.0
                                      };
                                      logsumexp(
                                        &mut expected_feature_count_sets.bulge_loop_length_counts
                                          [bulge_loop_len_2 - 1],
                                        bpap_4_2l,
                                      );
                                      // Count a 0x1 bulge loop.
                                      if bulge_loop_len_2 == 1 {
                                        let mismatch_2 = if loop_len_pair_2.0 == 0 {
                                          mismatch_pair_4.1
                                        } else {
                                          mismatch_pair_4.0
                                        };
                                        logsumexp(
                                          &mut expected_feature_count_sets
                                            .bulge_loop_0x1_length_counts[mismatch_2],
                                          bpap_4_2l,
                                        );
                                      }
                                    } else {
                                      // Count an interior loop length.
                                      logsumexp(
                                        &mut expected_feature_count_sets
                                          .interior_loop_length_counts
                                          [loop_len_pair_2.0 + loop_len_pair_2.1 - 2],
                                        bpap_4_2l,
                                      );
                                      let diff_2 = get_diff(loop_len_pair_2.0, loop_len_pair_2.1);
                                      if diff_2 == 0 {
                                        logsumexp(
                                          &mut expected_feature_count_sets
                                            .interior_loop_length_counts_symm
                                            [loop_len_pair_2.0 - 1],
                                          bpap_4_2l,
                                        );
                                      } else {
                                        logsumexp(
                                          &mut expected_feature_count_sets
                                            .interior_loop_length_counts_asymm[diff_2 - 1],
                                          bpap_4_2l,
                                        );
                                      }
                                      // Count a 1x1 interior loop.
                                      if loop_len_pair_2.0 == 1 && loop_len_pair_2.1 == 1 {
                                        let dict_min_mismatch_pair_4 =
                                          get_dict_min_mismatch_pair(&mismatch_pair_4);
                                        logsumexp(
                                          &mut expected_feature_count_sets
                                            .interior_loop_1x1_length_count_mat
                                            [dict_min_mismatch_pair_4.0]
                                            [dict_min_mismatch_pair_4.1],
                                          bpap_4_2l,
                                        );
                                      }
                                      // Count an explicit interior loop length pair.
                                      if loop_len_pair_2.0
                                        <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT
                                        && loop_len_pair_2.1
                                          <= CONSPROB_MAX_INTERIOR_LOOP_LEN_EXPLICIT
                                      {
                                        let dict_min_loop_len_pair_2 =
                                          get_dict_min_loop_len_pair(&loop_len_pair_2);
                                        logsumexp(
                                          &mut expected_feature_count_sets
                                            .interior_loop_length_count_mat_explicit
                                            [dict_min_loop_len_pair_2.0 - 1]
                                            [dict_min_loop_len_pair_2.1 - 1],
                                          bpap_4_2l,
                                        );
                                      }
                                    }
                                    // Count helix ends.
                                    logsumexp(
                                      &mut expected_feature_count_sets.helix_end_count_mat
                                        [base_pair_2.1][base_pair_2.0],
                                      bpap_4_2l,
                                    );
                                    logsumexp(
                                      &mut expected_feature_count_sets.helix_end_count_mat
                                        [base_pair_4.0][base_pair_4.1],
                                      bpap_4_2l,
                                    );
                                    // Count 2-loop terminal mismatches.
                                    logsumexp(
                                      &mut expected_feature_count_sets.terminal_mismatch_count_mat
                                        [base_pair_4.0][base_pair_4.1][mismatch_pair_4.0]
                                        [mismatch_pair_4.1],
                                      bpap_4_2l,
                                    );
                                    logsumexp(
                                      &mut expected_feature_count_sets.terminal_mismatch_count_mat
                                        [base_pair_2.1][base_pair_2.0][mismatch_pair_2.1]
                                        [mismatch_pair_2.0],
                                      bpap_4_2l,
                                    );
                                  }
                                }
                              }
                            }
                            None => {}
                          }
                        }
                      }
                      None => {}
                    }
                  }
                }
                let part_func_ratio = sta_part_func_mats.part_func_4d_mat_4_bpas_accessible_on_mls
                  [&pos_quadruple]
                  - part_func_4_bpa;
                for (pos_pair, forward_tmp_part_func_set_mat) in
                  &sta_part_func_mats.forward_tmp_part_func_set_mats_with_pos_pairs
                {
                  let &(u, v) = pos_pair;
                  if !(u < i && v < k) {
                    continue;
                  }
                  let ref forward_tmp_part_func_set_mat_decode = sta_part_func_mats
                    .forward_tmp_part_func_set_mats_with_pos_pairs_decode[pos_pair];
                  let pos_quadruple_2 = (u, j, v, l);
                  let mut forward_term = NEG_INFINITY;
                  let mut forward_term_4_align = forward_term;
                  let mut forward_term_2 = forward_term;
                  let mut forward_term_2_4_align = forward_term;
                  let pos_pair_2 = (i - T::one(), k - T::one());
                  match forward_tmp_part_func_set_mat_decode.get(&pos_pair_2) {
                    Some(part_funcs) => {
                      let part_func = part_funcs.part_func_4_bpas_on_mls;
                      logsumexp(&mut forward_term, part_func);
                      let part_func = part_funcs.part_func_on_sa_4_ml;
                      logsumexp(&mut forward_term_2, part_func);
                    }
                    None => {}
                  }
                  if train_score_params {
                    match forward_tmp_part_func_set_mat.get(&pos_pair_2) {
                      Some(part_funcs) => {
                        let part_func = part_funcs.part_func_4_bpas_on_mls;
                        let term = part_func + feature_score_sets.match_2_match_count;
                        logsumexp(&mut forward_term_4_align, term);
                        let part_func = part_funcs.part_func_on_sa_4_ml;
                        let term = part_func + feature_score_sets.match_2_match_count;
                        logsumexp(&mut forward_term_2_4_align, term);
                      }
                      None => {}
                    }
                  }
                  let mut part_func_4_ml = NEG_INFINITY;
                  match prob_coeff_mat_4_ml.get(&pos_quadruple_2) {
                    Some(prob_coeff_ml) => {
                      let prob_coeff_ml = prob_coeff_ml + part_func_ratio;
                      let term = prob_coeff_ml + forward_term;
                      logsumexp(&mut part_func_4_ml, term);
                      if train_score_params {
                        let prob = prob_coeff + prob_coeff_ml + forward_term_4_align;
                        logsumexp(&mut expected_feature_count_sets.match_2_match_count, prob);
                      }
                    }
                    None => {}
                  }
                  match prob_coeff_mat_4_ml_2.get(&pos_quadruple_2) {
                    Some(prob_coeff_ml_2) => {
                      let prob_coeff_ml_2 = prob_coeff_ml_2 + part_func_ratio;
                      let term = prob_coeff_ml_2 + forward_term_2;
                      logsumexp(&mut part_func_4_ml, term);
                      if train_score_params {
                        let prob = prob_coeff + prob_coeff_ml_2 + forward_term_2_4_align;
                        logsumexp(&mut expected_feature_count_sets.match_2_match_count, prob);
                      }
                    }
                    None => {}
                  }
                  if part_func_4_ml > NEG_INFINITY {
                    logsumexp(&mut sum, part_func_4_ml);
                    let bpap_4_ml = prob_coeff + part_func_4_ml;
                    if train_score_params {
                      // Count multi-loop terminal mismatches.
                      logsumexp(
                        &mut expected_feature_count_sets.left_dangle_count_mat[base_pair.1]
                          [base_pair.0][mismatch_pair.1],
                        bpap_4_ml,
                      );
                      logsumexp(
                        &mut expected_feature_count_sets.right_dangle_count_mat[base_pair.1]
                          [base_pair.0][mismatch_pair.0],
                        bpap_4_ml,
                      );
                      logsumexp(
                        &mut expected_feature_count_sets.left_dangle_count_mat[base_pair_2.1]
                          [base_pair_2.0][mismatch_pair_2.1],
                        bpap_4_ml,
                      );
                      logsumexp(
                        &mut expected_feature_count_sets.right_dangle_count_mat[base_pair_2.1]
                          [base_pair_2.0][mismatch_pair_2.0],
                        bpap_4_ml,
                      );
                      // Count helix ends.
                      logsumexp(
                        &mut expected_feature_count_sets.helix_end_count_mat[base_pair.1]
                          [base_pair.0],
                        bpap_4_ml,
                      );
                      logsumexp(
                        &mut expected_feature_count_sets.helix_end_count_mat[base_pair_2.1]
                          [base_pair_2.0],
                        bpap_4_ml,
                      );
                      // Count multi-loop closings.
                      logsumexp(
                        &mut expected_feature_count_sets.multi_loop_base_count,
                        (2. as Prob).ln() + bpap_4_ml,
                      );
                      // Count multi-loop closing basepairings.
                      logsumexp(
                        &mut expected_feature_count_sets.multi_loop_basepairing_count,
                        (2. as Prob).ln() + bpap_4_ml,
                      );
                      // Count multi-loop accessible basepairings.
                      logsumexp(
                        &mut expected_feature_count_sets.multi_loop_basepairing_count,
                        (2. as Prob).ln() + bpap_4_ml,
                      );
                    }
                  }
                }
                if sum > NEG_INFINITY {
                  sta_outside_part_func_4d_mat_4_bpas.insert(pos_quadruple, sum);
                  let bpap = prob_coeff + sum;
                  if produce_align_probs {
                    sta_prob_mats
                      .basepair_align_prob_mat
                      .insert(pos_quadruple, bpap);
                  }
                  if train_score_params {
                    // Count base pairs.
                    let dict_min_base_pair = get_dict_min_base_pair(&base_pair);
                    let dict_min_base_pair_2 = get_dict_min_base_pair(&base_pair_2);
                    logsumexp(
                      &mut expected_feature_count_sets.base_pair_count_mat[dict_min_base_pair.0]
                        [dict_min_base_pair.1],
                      bpap,
                    );
                    logsumexp(
                      &mut expected_feature_count_sets.base_pair_count_mat[dict_min_base_pair_2.0]
                        [dict_min_base_pair_2.1],
                      bpap,
                    );
                    // Count alignments.
                    let dict_min_align = get_dict_min_align(&(base_pair.0, base_pair_2.0));
                    logsumexp(
                      &mut expected_feature_count_sets.align_count_mat[dict_min_align.0]
                        [dict_min_align.1],
                      bpap,
                    );
                    let dict_min_align = get_dict_min_align(&(base_pair.1, base_pair_2.1));
                    logsumexp(
                      &mut expected_feature_count_sets.align_count_mat[dict_min_align.0]
                        [dict_min_align.1],
                      bpap,
                    );
                  }
                  match sta_prob_mats.bpp_mat_pair.0.get_mut(&(i, j)) {
                    Some(bpp) => {
                      logsumexp(bpp, bpap);
                    }
                    None => {
                      sta_prob_mats.bpp_mat_pair.0.insert((i, j), bpap);
                    }
                  }
                  match sta_prob_mats.bpp_mat_pair.1.get_mut(&(k, l)) {
                    Some(bpp) => {
                      logsumexp(bpp, bpap);
                    }
                    None => {
                      sta_prob_mats.bpp_mat_pair.1.insert((k, l), bpap);
                    }
                  }
                  logsumexp(&mut sta_prob_mats.bpp_mat_pair_2.0[long_i], bpap);
                  logsumexp(&mut sta_prob_mats.bpp_mat_pair_2.0[long_j], bpap);
                  logsumexp(&mut sta_prob_mats.bpp_mat_pair_2.1[long_k], bpap);
                  logsumexp(&mut sta_prob_mats.bpp_mat_pair_2.1[long_l], bpap);
                  let basepair_align_score = feature_score_sets.align_count_mat[base_pair.0]
                    [base_pair_2.0]
                    + feature_score_sets.align_count_mat[base_pair.1][base_pair_2.1];
                  let multi_loop_closing_basepairing_score =
                    bp_score_param_set_pair.0.multi_loop_closing_bp_scores[&(i, j)];
                  let multi_loop_closing_basepairing_score_2 =
                    bp_score_param_set_pair.1.multi_loop_closing_bp_scores[&(k, l)];
                  if train_score_params {
                    let mismatch_pair = (seq_pair.0[long_i + 1], seq_pair.0[long_j - 1]);
                    let mismatch_pair_2 = (seq_pair.1[long_k + 1], seq_pair.1[long_l - 1]);
                    let ref forward_tmp_part_func_set_mat =
                      sta_part_func_mats.forward_tmp_part_func_set_mats_with_pos_pairs[&(i, k)];
                    let ref forward_tmp_part_func_set_mat_decode = sta_part_func_mats
                      .forward_tmp_part_func_set_mats_with_pos_pairs_decode[&(i, k)];
                    if substr_len_1.to_usize().unwrap() - 2 <= CONSPROB_MAX_HAIRPIN_LOOP_LEN
                      && substr_len_2.to_usize().unwrap() - 2 <= CONSPROB_MAX_HAIRPIN_LOOP_LEN
                    {
                      let mut part_func_on_sa = NEG_INFINITY;
                      let mut part_func_on_sa_4_align = part_func_on_sa;
                      let pos_pair_2 = (j - T::one(), l - T::one());
                      match forward_tmp_part_func_set_mat_decode.get(&pos_pair_2) {
                        Some(part_funcs) => {
                          let part_func = part_funcs.part_func_on_sa;
                          logsumexp(&mut part_func_on_sa, part_func);
                        }
                        None => {}
                      }
                      match forward_tmp_part_func_set_mat.get(&pos_pair_2) {
                        Some(part_funcs) => {
                          let part_func = part_funcs.part_func_on_sa;
                          let term = part_func + feature_score_sets.match_2_match_count;
                          logsumexp(&mut part_func_on_sa_4_align, term);
                        }
                        None => {}
                      }
                      let hairpin_loop_score =
                        bp_score_param_set_pair.0.hairpin_loop_scores[&(i, j)];
                      let hairpin_loop_score_2 =
                        bp_score_param_set_pair.1.hairpin_loop_scores[&(k, l)];
                      let prob = sum - global_part_func
                        + part_func_on_sa_4_align
                        + hairpin_loop_score
                        + hairpin_loop_score_2
                        + basepair_align_score;
                      logsumexp(&mut expected_feature_count_sets.match_2_match_count, prob);
                      let bpap_4_hl = sum - global_part_func
                        + part_func_on_sa
                        + hairpin_loop_score
                        + hairpin_loop_score_2
                        + basepair_align_score;
                      logsumexp(
                        &mut expected_feature_count_sets.hairpin_loop_length_counts
                          [long_j - long_i - 1],
                        bpap_4_hl,
                      );
                      logsumexp(
                        &mut expected_feature_count_sets.hairpin_loop_length_counts
                          [long_l - long_k - 1],
                        bpap_4_hl,
                      );
                      logsumexp(
                        &mut expected_feature_count_sets.terminal_mismatch_count_mat[base_pair.0]
                          [base_pair.1][mismatch_pair.0][mismatch_pair.1],
                        bpap_4_hl,
                      );
                      logsumexp(
                        &mut expected_feature_count_sets.terminal_mismatch_count_mat[base_pair_2.0]
                          [base_pair_2.1][mismatch_pair_2.0][mismatch_pair_2.1],
                        bpap_4_hl,
                      );
                      // Count helix ends.
                      logsumexp(
                        &mut expected_feature_count_sets.helix_end_count_mat[base_pair.0]
                          [base_pair.1],
                        bpap_4_hl,
                      );
                      logsumexp(
                        &mut expected_feature_count_sets.helix_end_count_mat[base_pair_2.0]
                          [base_pair_2.1],
                        bpap_4_hl,
                      );
                    }
                    let mut part_func_4_ml = NEG_INFINITY;
                    let mut part_func_4_ml_4_align = part_func_4_ml;
                    match forward_tmp_part_func_set_mat_decode.get(&pos_pair_2) {
                      Some(part_funcs) => {
                        let part_func = part_funcs.part_func_4_ml;
                        logsumexp(&mut part_func_4_ml, part_func);
                      }
                      None => {}
                    }
                    match forward_tmp_part_func_set_mat.get(&pos_pair_2) {
                      Some(part_funcs) => {
                        let part_func = part_funcs.part_func_4_ml;
                        let term = part_func + feature_score_sets.match_2_match_count;
                        logsumexp(&mut part_func_4_ml_4_align, term);
                      }
                      None => {}
                    }
                    let prob = sum - global_part_func
                      + part_func_4_ml_4_align
                      + multi_loop_closing_basepairing_score
                      + multi_loop_closing_basepairing_score_2
                      + basepair_align_score;
                    logsumexp(&mut expected_feature_count_sets.match_2_match_count, prob);
                    let bpap_4_ml = sum - global_part_func
                      + part_func_4_ml
                      + multi_loop_closing_basepairing_score
                      + multi_loop_closing_basepairing_score_2
                      + basepair_align_score;
                    // Count multi-loop terminal mismatches.
                    logsumexp(
                      &mut expected_feature_count_sets.left_dangle_count_mat[base_pair.0]
                        [base_pair.1][mismatch_pair.0],
                      bpap_4_ml,
                    );
                    logsumexp(
                      &mut expected_feature_count_sets.right_dangle_count_mat[base_pair.0]
                        [base_pair.1][mismatch_pair.1],
                      bpap_4_ml,
                    );
                    logsumexp(
                      &mut expected_feature_count_sets.left_dangle_count_mat[base_pair_2.0]
                        [base_pair_2.1][mismatch_pair_2.0],
                      bpap_4_ml,
                    );
                    logsumexp(
                      &mut expected_feature_count_sets.right_dangle_count_mat[base_pair_2.0]
                        [base_pair_2.1][mismatch_pair_2.1],
                      bpap_4_ml,
                    );
                    // Count helix ends.
                    logsumexp(
                      &mut expected_feature_count_sets.helix_end_count_mat[base_pair.0]
                        [base_pair.1],
                      bpap_4_ml,
                    );
                    logsumexp(
                      &mut expected_feature_count_sets.helix_end_count_mat[base_pair_2.0]
                        [base_pair_2.1],
                      bpap_4_ml,
                    );
                  }
                  let coefficient = sum
                    + basepair_align_score
                    + multi_loop_closing_basepairing_score
                    + multi_loop_closing_basepairing_score_2;
                  let ref backward_tmp_part_func_set_mat_decode = sta_part_func_mats
                    .backward_tmp_part_func_set_mats_with_pos_pairs_decode[&(j, l)];
                  for pos_pair in align_prob_mat.keys() {
                    let &(u, v) = pos_pair;
                    if !(i < u && u < j && k < v && v < l) {
                      continue;
                    }
                    let mut backward_term = NEG_INFINITY;
                    let mut backward_term_2 = backward_term;
                    let pos_pair_2 = (u + T::one(), v + T::one());
                    match backward_tmp_part_func_set_mat_decode.get(&pos_pair_2) {
                      Some(part_funcs) => {
                        let part_func = part_funcs.part_func_on_mls;
                        logsumexp(&mut backward_term, part_func);
                        let part_func = part_funcs.part_func_4_bpas_on_mls;
                        logsumexp(&mut backward_term_2, part_func);
                      }
                      None => {}
                    }
                    let pos_quadruple_2 = (i, u, k, v);
                    let prob_coeff_4_ml = coefficient + backward_term;
                    match prob_coeff_mat_4_ml.get_mut(&pos_quadruple_2) {
                      Some(x) => {
                        logsumexp(x, prob_coeff_4_ml);
                      }
                      None => {
                        prob_coeff_mat_4_ml.insert(pos_quadruple_2, prob_coeff_4_ml);
                      }
                    }
                    let prob_coeff_4_ml_2 = coefficient + backward_term_2;
                    match prob_coeff_mat_4_ml_2.get_mut(&pos_quadruple_2) {
                      Some(x) => {
                        logsumexp(x, prob_coeff_4_ml_2);
                      }
                      None => {
                        prob_coeff_mat_4_ml_2.insert(pos_quadruple_2, prob_coeff_4_ml_2);
                      }
                    }
                  }
                }
              }
              None => {}
            }
          }
        }
        None => {}
      }
    }
  }
  for bpp in sta_prob_mats.bpp_mat_pair.0.values_mut() {
    *bpp = expf(*bpp);
  }
  for bpp in sta_prob_mats.bpp_mat_pair.1.values_mut() {
    *bpp = expf(*bpp);
  }
  let need_twoloop_part_funcs = produce_align_probs || train_score_params;
  let need_indel_info = produce_struct_profs || train_score_params;
  if produce_struct_profs || need_twoloop_part_funcs {
    let mut upp_mat_pair_4_el_range =
      (SparseProbMat::<T>::default(), SparseProbMat::<T>::default());
    let mut upp_mat_pair_4_hl_range =
      (SparseProbMat::<T>::default(), SparseProbMat::<T>::default());
    let mut upp_mat_pair_range = (SparseProbMat::<T>::default(), SparseProbMat::<T>::default());
    for u in range(T::zero(), seq_len_pair.0 - T::one()) {
      let long_u = u.to_usize().unwrap();
      let base = seq_pair.0[long_u];
      for v in range(T::zero(), seq_len_pair.1 - T::one()) {
        let pos_pair = (u, v);
        let long_v = v.to_usize().unwrap();
        let base_2 = seq_pair.1[long_v];
        let pos_pair_2 = (u + T::one(), v + T::one());
        let long_pos_pair_2 = (
          pos_pair_2.0.to_usize().unwrap(),
          pos_pair_2.1.to_usize().unwrap(),
        );
        let dict_min_align = get_dict_min_align(&(base, base_2));
        if align_prob_mat.contains_key(&pos_pair) {
          let pos_pair_4_loop_align = (u - T::one(), v - T::one());
          let mut loop_align_prob_4_el = NEG_INFINITY;
          let loop_align_score = feature_score_sets.align_count_mat[base][base_2]
            + 2. * feature_score_sets.external_loop_accessible_baseunpairing_count;
          let mut backward_term = NEG_INFINITY;
          match sta_part_func_mats
            .backward_part_func_mat_4_external_loop_decode
            .get(&pos_pair_2)
          {
            Some(&part_func) => {
              logsumexp(&mut backward_term, part_func);
            }
            None => {}
          }
          match sta_part_func_mats
            .forward_part_func_mat_4_external_loop_decode
            .get(&pos_pair_4_loop_align)
          {
            Some(&forward_term) => {
              let term = loop_align_score + forward_term + backward_term - global_part_func;
              logsumexp(&mut loop_align_prob_4_el, term);
            }
            None => {}
          }
          if produce_struct_profs {
            logsumexp(
              &mut sta_prob_mats.upp_mat_pair_4_el.0[long_u],
              loop_align_prob_4_el,
            );
            logsumexp(
              &mut sta_prob_mats.upp_mat_pair_4_el.1[long_v],
              loop_align_prob_4_el,
            );
          }
          if produce_align_probs {
            match sta_prob_mats.loop_align_prob_mat.get_mut(&pos_pair) {
              Some(loop_align_prob) => {
                logsumexp(loop_align_prob, loop_align_prob_4_el);
              }
              None => {
                sta_prob_mats
                  .loop_align_prob_mat
                  .insert(pos_pair, loop_align_prob_4_el);
              }
            }
          }
          if train_score_params {
            logsumexp(
              &mut expected_feature_count_sets.align_count_mat[dict_min_align.0][dict_min_align.1],
              loop_align_prob_4_el,
            );
            logsumexp(
              &mut expected_feature_count_sets.external_loop_accessible_baseunpairing_count,
              (2. as Prob).ln() + loop_align_prob_4_el,
            );
            match sta_part_func_mats
              .forward_part_func_mat_4_external_loop
              .get(&pos_pair_4_loop_align)
            {
              Some(&part_func) => {
                let is_begin = pos_pair_4_loop_align == leftmost_pos_pair;
                let forward_term = part_func
                  + if is_begin {
                    feature_score_sets.init_match_count
                  } else {
                    feature_score_sets.match_2_match_count
                  };
                let term = loop_align_score + forward_term + backward_term - global_part_func;
                if is_begin {
                  logsumexp(&mut expected_feature_count_sets.init_match_count, term);
                } else {
                  logsumexp(&mut expected_feature_count_sets.match_2_match_count, term);
                }
              }
              None => {}
            }
          }
        }
        if need_indel_info {
          match sta_part_func_mats
            .forward_part_func_mat_4_external_loop
            .get(&pos_pair)
          {
            Some(&part_func) => {
              let is_begin = pos_pair == leftmost_pos_pair;
              let forward_term = part_func
                + if is_begin {
                  feature_score_sets.init_insert_count
                } else {
                  feature_score_sets.match_2_insert_count
                };
              match matchable_pos_sets_1.get(&pos_pair_2.0) {
                Some(matchable_poss) => {
                  for &x in matchable_poss {
                    if x <= pos_pair_2.1 {
                      continue;
                    }
                    let pos_pair_3 = (pos_pair_2.0, x);
                    match sta_part_func_mats
                      .backward_part_func_mat_4_external_loop
                      .get(&pos_pair_3)
                    {
                      Some(&backward_term) => {
                        let long_x = x.to_usize().unwrap();
                        let is_end = pos_pair_3 == rightmost_pos_pair;
                        let insert_score_range = insert_score_range_sets.insert_scores_range_4_el_2
                          [long_pos_pair_2.1][long_x - 1]
                          + if is_end {
                            0.
                          } else {
                            feature_score_sets.match_2_insert_count
                          };
                        let term =
                          forward_term + insert_score_range + backward_term - global_part_func;
                        if train_score_params {
                          if is_begin {
                            logsumexp(&mut expected_feature_count_sets.init_insert_count, term);
                          } else {
                            logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                          }
                          if !is_end {
                            logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                          }
                          logsumexp(
                            &mut expected_feature_count_sets
                              .external_loop_accessible_baseunpairing_count,
                            ((long_x - long_pos_pair_2.1) as Prob).ln() + term,
                          );
                          if pos_pair_2.1 < x - T::one() {
                            logsumexp(
                              &mut expected_feature_count_sets.insert_extend_count,
                              ((long_x - long_pos_pair_2.1 - 1) as Prob).ln() + term,
                            );
                          }
                        }
                        let pos_pair_4 = (pos_pair_2.1, x - T::one());
                        if produce_struct_profs {
                          match upp_mat_pair_4_el_range.1.get_mut(&pos_pair_4) {
                            Some(upp) => {
                              logsumexp(upp, term);
                            }
                            None => {
                              upp_mat_pair_4_el_range.1.insert(pos_pair_4, term);
                            }
                          }
                        }
                        if train_score_params {
                          match upp_mat_pair_range.1.get_mut(&pos_pair_4) {
                            Some(upp) => {
                              logsumexp(upp, term);
                            }
                            None => {
                              upp_mat_pair_range.1.insert(pos_pair_4, term);
                            }
                          }
                        }
                      }
                      None => {}
                    }
                  }
                }
                None => {}
              }
              match matchable_pos_sets_2.get(&pos_pair_2.1) {
                Some(matchable_poss) => {
                  for &x in matchable_poss {
                    if x <= pos_pair_2.0 {
                      continue;
                    }
                    let pos_pair_3 = (x, pos_pair_2.1);
                    match sta_part_func_mats
                      .backward_part_func_mat_4_external_loop
                      .get(&pos_pair_3)
                    {
                      Some(&backward_term) => {
                        let long_x = x.to_usize().unwrap();
                        let is_end = pos_pair_3 == rightmost_pos_pair;
                        let insert_score_range = insert_score_range_sets.insert_scores_range_4_el
                          [long_pos_pair_2.0][long_x - 1]
                          + if is_end {
                            0.
                          } else {
                            feature_score_sets.match_2_insert_count
                          };
                        let term =
                          forward_term + insert_score_range + backward_term - global_part_func;
                        if train_score_params {
                          if is_begin {
                            logsumexp(&mut expected_feature_count_sets.init_insert_count, term);
                          } else {
                            logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                          }
                          if !is_end {
                            logsumexp(&mut expected_feature_count_sets.match_2_insert_count, term);
                          }
                          logsumexp(
                            &mut expected_feature_count_sets
                              .external_loop_accessible_baseunpairing_count,
                            ((long_x - long_pos_pair_2.0) as Prob).ln() + term,
                          );
                          if pos_pair_2.0 < x - T::one() {
                            logsumexp(
                              &mut expected_feature_count_sets.insert_extend_count,
                              ((long_x - long_pos_pair_2.0 - 1) as Prob).ln() + term,
                            );
                          }
                        }
                        let pos_pair_4 = (pos_pair_2.0, x - T::one());
                        if produce_struct_profs {
                          match upp_mat_pair_4_el_range.0.get_mut(&pos_pair_4) {
                            Some(upp) => {
                              logsumexp(upp, term);
                            }
                            None => {
                              upp_mat_pair_4_el_range.0.insert(pos_pair_4, term);
                            }
                          }
                        }
                        if train_score_params {
                          match upp_mat_pair_range.0.get_mut(&pos_pair_4) {
                            Some(upp) => {
                              logsumexp(upp, term);
                            }
                            None => {
                              upp_mat_pair_range.0.insert(pos_pair_4, term);
                            }
                          }
                        }
                      }
                      None => {}
                    }
                  }
                }
                None => {}
              }
            }
            None => {}
          }
        }
      }
    }
    for (pos_quadruple, &part_func_4_bpa) in &sta_outside_part_func_4d_mat_4_bpas {
      let (i, j, k, l) = *pos_quadruple;
      let (long_i, long_j, long_k, long_l) = (
        i.to_usize().unwrap(),
        j.to_usize().unwrap(),
        k.to_usize().unwrap(),
        l.to_usize().unwrap(),
      );
      let base_pair = (seq_pair.0[long_i], seq_pair.0[long_j]);
      let base_pair_2 = (seq_pair.1[long_k], seq_pair.1[long_l]);
      let hairpin_loop_score =
        if j - i - T::one() <= T::from_usize(CONSPROB_MAX_HAIRPIN_LOOP_LEN).unwrap() {
          bp_score_param_set_pair.0.hairpin_loop_scores[&(i, j)]
        } else {
          NEG_INFINITY
        };
      let hairpin_loop_score_2 =
        if l - k - T::one() <= T::from_usize(CONSPROB_MAX_HAIRPIN_LOOP_LEN).unwrap() {
          bp_score_param_set_pair.1.hairpin_loop_scores[&(k, l)]
        } else {
          NEG_INFINITY
        };
      let multi_loop_closing_basepairing_score =
        bp_score_param_set_pair.0.multi_loop_closing_bp_scores[&(i, j)];
      let multi_loop_closing_basepairing_score_2 =
        bp_score_param_set_pair.1.multi_loop_closing_bp_scores[&(k, l)];
      let basepair_align_score = feature_score_sets.align_count_mat[base_pair.0][base_pair_2.0]
        + feature_score_sets.align_count_mat[base_pair.1][base_pair_2.1];
      let prob_coeff = part_func_4_bpa - global_part_func + basepair_align_score;
      let ref forward_tmp_part_func_set_mat =
        sta_part_func_mats.forward_tmp_part_func_set_mats_with_pos_pairs[&(i, k)];
      let ref forward_tmp_part_func_set_mat_decode =
        sta_part_func_mats.forward_tmp_part_func_set_mats_with_pos_pairs_decode[&(i, k)];
      let ref backward_tmp_part_func_set_mat =
        sta_part_func_mats.backward_tmp_part_func_set_mats_with_pos_pairs[&(j, l)];
      let ref backward_tmp_part_func_set_mat_decode =
        sta_part_func_mats.backward_tmp_part_func_set_mats_with_pos_pairs_decode[&(j, l)];
      let (forward_part_func_mat_4_2loop, forward_part_func_mat_4_2loop_decode) =
        if need_twoloop_part_funcs {
          get_part_func_mat_2loop(
            seq_pair,
            feature_score_sets,
            align_prob_mat,
            &pos_quadruple,
            &sta_part_func_mats,
            true,
            forward_pos_pair_mat_set,
            bp_score_param_set_pair,
            insert_score_range_sets,
            matchable_pos_sets_1,
            matchable_pos_sets_2,
          )
        } else {
          (
            SparsePartFuncMat::<T>::default(),
            SparsePartFuncMat::<T>::default(),
          )
        };
      let (backward_part_func_mat_4_2loop, backward_part_func_mat_4_2loop_decode) =
        if need_twoloop_part_funcs {
          get_part_func_mat_2loop(
            seq_pair,
            feature_score_sets,
            align_prob_mat,
            &pos_quadruple,
            &sta_part_func_mats,
            false,
            backward_pos_pair_mat_set,
            bp_score_param_set_pair,
            insert_score_range_sets,
            matchable_pos_sets_1,
            matchable_pos_sets_2,
          )
        } else {
          (
            SparsePartFuncMat::<T>::default(),
            SparsePartFuncMat::<T>::default(),
          )
        };
      if train_score_params {
        let pos_pair_2 = (j - T::one(), l - T::one());
        match forward_part_func_mat_4_2loop.get(&pos_pair_2) {
          Some(&forward_term) => {
            let term = prob_coeff + forward_term + feature_score_sets.match_2_match_count;
            logsumexp(&mut expected_feature_count_sets.match_2_match_count, term);
          }
          None => {}
        }
      }
      for u in range_inclusive(i, j) {
        let long_u = u.to_usize().unwrap();
        let base = seq_pair.0[long_u];
        for v in range_inclusive(k, l) {
          let pos_pair = (u, v);
          let long_v = v.to_usize().unwrap();
          let base_2 = seq_pair.1[long_v];
          let pos_pair_2 = (u + T::one(), v + T::one());
          let long_pos_pair_2 = (
            pos_pair_2.0.to_usize().unwrap(),
            pos_pair_2.1.to_usize().unwrap(),
          );
          let dict_min_align = get_dict_min_align(&(base, base_2));
          let mut backward_term_on_sa = NEG_INFINITY;
          let mut backward_term_4_ml = backward_term_on_sa;
          let mut backward_term_4_bpas_on_mls = backward_term_on_sa;
          let mut backward_term_on_mls = backward_term_on_sa;
          let mut backward_term_4_2loop = backward_term_on_sa;
          match backward_tmp_part_func_set_mat_decode.get(&pos_pair_2) {
            Some(part_funcs) => {
              let part_func = part_funcs.part_func_on_sa;
              logsumexp(&mut backward_term_on_sa, part_func);
              if need_twoloop_part_funcs {
                let part_func = part_funcs.part_func_4_ml;
                logsumexp(&mut backward_term_4_ml, part_func);
                let part_func = part_funcs.part_func_4_bpas_on_mls;
                logsumexp(&mut backward_term_4_bpas_on_mls, part_func);
                let part_func = part_funcs.part_func_on_mls;
                logsumexp(&mut backward_term_on_mls, part_func);
              }
            }
            None => {}
          }
          if need_twoloop_part_funcs {
            match backward_part_func_mat_4_2loop_decode.get(&pos_pair_2) {
              Some(&part_func) => {
                logsumexp(&mut backward_term_4_2loop, part_func);
              }
              None => {}
            }
          }
          let prob_coeff_4_hl = prob_coeff + hairpin_loop_score + hairpin_loop_score_2;
          let prob_coeff_4_ml = prob_coeff
            + multi_loop_closing_basepairing_score
            + multi_loop_closing_basepairing_score_2;
          if align_prob_mat.contains_key(&pos_pair) {
            let pos_pair_4_loop_align = (u - T::one(), v - T::one());
            let loop_align_score = feature_score_sets.align_count_mat[base][base_2];
            let loop_align_score_ml =
              loop_align_score + 2. * feature_score_sets.multi_loop_accessible_baseunpairing_count;
            let mut loop_align_prob_4_hairpin_loop = NEG_INFINITY;
            let mut loop_align_prob_4_multi_loop = loop_align_prob_4_hairpin_loop;
            let mut loop_align_prob_4_2loop = loop_align_prob_4_hairpin_loop;
            match forward_tmp_part_func_set_mat_decode.get(&pos_pair_4_loop_align) {
              Some(part_funcs) => {
                let part_func = part_funcs.part_func_on_sa;
                let term = prob_coeff_4_hl + loop_align_score + part_func + backward_term_on_sa;
                logsumexp(&mut loop_align_prob_4_hairpin_loop, term);
                if need_twoloop_part_funcs {
                  let part_func = part_funcs.part_func_on_sa_4_ml;
                  let term = prob_coeff_4_ml + loop_align_score_ml + part_func + backward_term_4_ml;
                  logsumexp(&mut loop_align_prob_4_multi_loop, term);
                  let part_func = part_funcs.part_func_4_first_bpas_on_mls;
                  let term =
                    prob_coeff_4_ml + loop_align_score_ml + part_func + backward_term_4_bpas_on_mls;
                  logsumexp(&mut loop_align_prob_4_multi_loop, term);
                  let part_func = part_funcs.part_func_4_ml;
                  let term =
                    prob_coeff_4_ml + loop_align_score_ml + part_func + backward_term_on_mls;
                  logsumexp(&mut loop_align_prob_4_multi_loop, term);
                  let part_func = part_funcs.part_func_on_sa;
                  let term = prob_coeff + loop_align_score + part_func + backward_term_4_2loop;
                  logsumexp(&mut loop_align_prob_4_2loop, term);
                }
              }
              None => {}
            }
            if produce_struct_profs {
              logsumexp(
                &mut sta_prob_mats.upp_mat_pair_4_hl.0[long_u],
                loop_align_prob_4_hairpin_loop,
              );
              logsumexp(
                &mut sta_prob_mats.upp_mat_pair_4_hl.1[long_v],
                loop_align_prob_4_hairpin_loop,
              );
            }
            if produce_align_probs || train_score_params {
              match forward_part_func_mat_4_2loop_decode.get(&pos_pair_4_loop_align) {
                Some(&part_func) => {
                  let term = prob_coeff + loop_align_score + part_func + backward_term_on_sa;
                  logsumexp(&mut loop_align_prob_4_2loop, term);
                }
                None => {}
              }
              let mut prob = NEG_INFINITY;
              logsumexp(&mut prob, loop_align_prob_4_hairpin_loop);
              logsumexp(&mut prob, loop_align_prob_4_multi_loop);
              logsumexp(&mut prob, loop_align_prob_4_2loop);
              if produce_align_probs {
                match sta_prob_mats.loop_align_prob_mat.get_mut(&pos_pair) {
                  Some(loop_align_prob) => {
                    logsumexp(loop_align_prob, prob);
                  }
                  None => {
                    sta_prob_mats.loop_align_prob_mat.insert(pos_pair, prob);
                  }
                }
              }
              if train_score_params {
                logsumexp(
                  &mut expected_feature_count_sets.align_count_mat[dict_min_align.0]
                    [dict_min_align.1],
                  prob,
                );
                logsumexp(
                  &mut expected_feature_count_sets.multi_loop_accessible_baseunpairing_count,
                  (2. as Prob).ln() + loop_align_prob_4_multi_loop,
                );
              }
            }
            match forward_tmp_part_func_set_mat.get(&pos_pair_4_loop_align) {
              Some(part_funcs) => {
                let part_func = part_funcs.part_func_on_sa;
                let term = prob_coeff_4_hl
                  + loop_align_score
                  + part_func
                  + feature_score_sets.match_2_match_count
                  + backward_term_on_sa;
                if train_score_params {
                  logsumexp(&mut expected_feature_count_sets.match_2_match_count, term);
                }
                if need_twoloop_part_funcs {
                  let part_func = part_funcs.part_func_on_sa_4_ml;
                  let term = prob_coeff_4_ml
                    + loop_align_score_ml
                    + part_func
                    + feature_score_sets.match_2_match_count
                    + backward_term_4_ml;
                  if train_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_match_count, term);
                  }
                  let part_func = part_funcs.part_func_4_first_bpas_on_mls;
                  let term = prob_coeff_4_ml
                    + loop_align_score_ml
                    + part_func
                    + feature_score_sets.match_2_match_count
                    + backward_term_4_bpas_on_mls;
                  if train_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_match_count, term);
                  }
                  let part_func = part_funcs.part_func_4_ml;
                  let term = prob_coeff_4_ml
                    + loop_align_score_ml
                    + part_func
                    + feature_score_sets.match_2_match_count
                    + backward_term_on_mls;
                  if train_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_match_count, term);
                  }
                  let part_func = part_funcs.part_func_on_sa;
                  let term = prob_coeff
                    + loop_align_score
                    + part_func
                    + feature_score_sets.match_2_match_count
                    + backward_term_4_2loop;
                  if train_score_params {
                    logsumexp(&mut expected_feature_count_sets.match_2_match_count, term);
                  }
                }
              }
              None => {}
            }
            if need_indel_info {
              match forward_tmp_part_func_set_mat.get(&pos_pair) {
                Some(part_funcs) => {
                  let forward_term_on_sa =
                    part_funcs.part_func_on_sa + feature_score_sets.match_2_insert_count;
                  let forward_term_on_sa_4_ml =
                    part_funcs.part_func_on_sa_4_ml + feature_score_sets.match_2_insert_count;
                  let forward_term_4_first_bpas_on_mls = part_funcs.part_func_4_first_bpas_on_mls
                    + feature_score_sets.match_2_insert_count;
                  let forward_term_4_ml =
                    part_funcs.part_func_4_ml + feature_score_sets.match_2_insert_count;
                  match matchable_pos_sets_1.get(&pos_pair_2.0) {
                    Some(matchable_poss) => {
                      for &x in matchable_poss {
                        if x <= pos_pair_2.1 {
                          continue;
                        }
                        let mut insert_prob = NEG_INFINITY;
                        let mut insert_prob_ml = insert_prob;
                        let long_x = x.to_usize().unwrap();
                        let insert_score_range = insert_score_range_sets.insert_scores_range_2
                          [long_pos_pair_2.1][long_x - 1]
                          + feature_score_sets.match_2_insert_count;
                        match backward_tmp_part_func_set_mat.get(&(pos_pair_2.0, x)) {
                          Some(part_funcs) => {
                            let backward_term = part_funcs.part_func_on_sa;
                            let term = prob_coeff_4_hl
                              + forward_term_on_sa
                              + insert_score_range
                              + backward_term;
                            logsumexp(&mut insert_prob, term);
                            if produce_struct_profs {
                              let pos_pair_4 = (pos_pair_2.1, x - T::one());
                              match upp_mat_pair_4_hl_range.1.get_mut(&pos_pair_4) {
                                Some(upp) => {
                                  logsumexp(upp, term);
                                }
                                None => {
                                  upp_mat_pair_4_hl_range.1.insert(pos_pair_4, term);
                                }
                              }
                            }
                            if train_score_params {
                              let insert_score_range_4_ml = insert_score_range_sets
                                .insert_scores_range_4_ml_2[long_pos_pair_2.1][long_x - 1]
                                + feature_score_sets.match_2_insert_count;
                              let backward_term = part_funcs.part_func_4_ml;
                              let term = prob_coeff_4_ml
                                + forward_term_on_sa_4_ml
                                + insert_score_range_4_ml
                                + backward_term;
                              logsumexp(&mut insert_prob_ml, term);
                              let backward_term = part_funcs.part_func_4_bpas_on_mls;
                              let term = prob_coeff_4_ml
                                + forward_term_4_first_bpas_on_mls
                                + insert_score_range_4_ml
                                + backward_term;
                              logsumexp(&mut insert_prob_ml, term);
                              let backward_term = part_funcs.part_func_on_mls;
                              let term = prob_coeff_4_ml
                                + forward_term_4_ml
                                + insert_score_range_4_ml
                                + backward_term;
                              logsumexp(&mut insert_prob_ml, term);
                              logsumexp(&mut insert_prob, insert_prob_ml);
                            }
                          }
                          None => {}
                        }
                        if train_score_params {
                          match backward_part_func_mat_4_2loop.get(&(pos_pair_2.0, x)) {
                            Some(&backward_term) => {
                              let term = prob_coeff
                                + forward_term_on_sa
                                + insert_score_range
                                + backward_term;
                              logsumexp(&mut insert_prob, term);
                            }
                            None => {}
                          }
                          logsumexp(
                            &mut expected_feature_count_sets.match_2_insert_count,
                            (2. as Prob).ln() + insert_prob,
                          );
                          let pos_pair_4 = (pos_pair_2.1, x - T::one());
                          match upp_mat_pair_range.1.get_mut(&pos_pair_4) {
                            Some(upp) => {
                              logsumexp(upp, insert_prob);
                            }
                            None => {
                              upp_mat_pair_range.1.insert(pos_pair_4, insert_prob);
                            }
                          }
                          logsumexp(
                            &mut expected_feature_count_sets
                              .multi_loop_accessible_baseunpairing_count,
                            ((long_x - long_pos_pair_2.1) as Prob).ln() + insert_prob_ml,
                          );
                          if pos_pair_2.1 < x - T::one() {
                            logsumexp(
                              &mut expected_feature_count_sets.insert_extend_count,
                              ((long_x - long_pos_pair_2.1 - 1) as Prob).ln() + insert_prob,
                            );
                          }
                        }
                      }
                    }
                    None => {}
                  }
                  match matchable_pos_sets_2.get(&pos_pair_2.1) {
                    Some(matchable_poss) => {
                      for &x in matchable_poss {
                        if x <= pos_pair_2.0 {
                          continue;
                        }
                        let mut insert_prob = NEG_INFINITY;
                        let mut insert_prob_ml = insert_prob;
                        let long_x = x.to_usize().unwrap();
                        let insert_score_range = insert_score_range_sets.insert_scores_range
                          [long_pos_pair_2.0][long_x - 1]
                          + feature_score_sets.match_2_insert_count;
                        match backward_tmp_part_func_set_mat.get(&(x, pos_pair_2.1)) {
                          Some(part_funcs) => {
                            let backward_term = part_funcs.part_func_on_sa;
                            let term = prob_coeff_4_hl
                              + forward_term_on_sa
                              + insert_score_range
                              + backward_term;
                            logsumexp(&mut insert_prob, term);
                            if produce_struct_profs {
                              let pos_pair_4 = (pos_pair_2.0, x - T::one());
                              match upp_mat_pair_4_hl_range.0.get_mut(&pos_pair_4) {
                                Some(upp) => {
                                  logsumexp(upp, term);
                                }
                                None => {
                                  upp_mat_pair_4_hl_range.0.insert(pos_pair_4, term);
                                }
                              }
                            }
                            if train_score_params {
                              let insert_score_range_4_ml = insert_score_range_sets
                                .insert_scores_range_4_ml[long_pos_pair_2.0][long_x - 1]
                                + feature_score_sets.match_2_insert_count;
                              let backward_term = part_funcs.part_func_4_ml;
                              let term = prob_coeff_4_ml
                                + forward_term_on_sa_4_ml
                                + insert_score_range_4_ml
                                + backward_term;
                              logsumexp(&mut insert_prob_ml, term);
                              let backward_term = part_funcs.part_func_4_bpas_on_mls;
                              let term = prob_coeff_4_ml
                                + forward_term_4_first_bpas_on_mls
                                + insert_score_range_4_ml
                                + backward_term;
                              logsumexp(&mut insert_prob_ml, term);
                              let backward_term = part_funcs.part_func_on_mls;
                              let term = prob_coeff_4_ml
                                + forward_term_4_ml
                                + insert_score_range_4_ml
                                + backward_term;
                              logsumexp(&mut insert_prob_ml, term);
                              logsumexp(&mut insert_prob, insert_prob_ml);
                            }
                          }
                          None => {}
                        }
                        match backward_part_func_mat_4_2loop.get(&(x, pos_pair_2.1)) {
                          Some(&backward_term) => {
                            let term =
                              prob_coeff + forward_term_on_sa + insert_score_range + backward_term;
                            logsumexp(&mut insert_prob, term);
                          }
                          None => {}
                        }
                        if train_score_params {
                          logsumexp(
                            &mut expected_feature_count_sets.match_2_insert_count,
                            (2. as Prob).ln() + insert_prob,
                          );
                          let pos_pair_4 = (pos_pair_2.0, x - T::one());
                          match upp_mat_pair_range.0.get_mut(&pos_pair_4) {
                            Some(upp) => {
                              logsumexp(upp, insert_prob);
                            }
                            None => {
                              upp_mat_pair_range.0.insert(pos_pair_4, insert_prob);
                            }
                          }
                          logsumexp(
                            &mut expected_feature_count_sets
                              .multi_loop_accessible_baseunpairing_count,
                            ((long_x - long_pos_pair_2.0) as Prob).ln() + insert_prob_ml,
                          );
                          if pos_pair_2.0 < x - T::one() {
                            logsumexp(
                              &mut expected_feature_count_sets.insert_extend_count,
                              ((long_x - long_pos_pair_2.0 - 1) as Prob).ln() + insert_prob,
                            );
                          }
                        }
                      }
                    }
                    None => {}
                  }
                }
                None => {}
              }
              if train_score_params {
                match forward_part_func_mat_4_2loop.get(&pos_pair) {
                  Some(&forward_term) => {
                    match matchable_pos_sets_1.get(&pos_pair_2.0) {
                      Some(matchable_poss) => {
                        for &x in matchable_poss {
                          if x <= pos_pair_2.1 {
                            continue;
                          }
                          let mut insert_prob = NEG_INFINITY;
                          let long_x = x.to_usize().unwrap();
                          let insert_score_range = insert_score_range_sets.insert_scores_range_2
                            [long_pos_pair_2.1][long_x - 1]
                            + feature_score_sets.match_2_insert_count;
                          match backward_tmp_part_func_set_mat.get(&(pos_pair_2.0, x)) {
                            Some(part_funcs) => {
                              let backward_term = part_funcs.part_func_on_sa;
                              let term =
                                prob_coeff + forward_term + insert_score_range + backward_term;
                              logsumexp(&mut insert_prob, term);
                              logsumexp(
                                &mut expected_feature_count_sets.match_2_insert_count,
                                (2. as Prob).ln() + insert_prob,
                              );
                              let pos_pair_4 = (pos_pair_2.1, x - T::one());
                              match upp_mat_pair_range.1.get_mut(&pos_pair_4) {
                                Some(upp) => {
                                  logsumexp(upp, insert_prob);
                                }
                                None => {
                                  upp_mat_pair_range.1.insert(pos_pair_4, insert_prob);
                                }
                              }
                              if pos_pair_2.1 < x - T::one() {
                                logsumexp(
                                  &mut expected_feature_count_sets.insert_extend_count,
                                  ((long_x - long_pos_pair_2.1 - 1) as Prob).ln() + insert_prob,
                                );
                              }
                            }
                            None => {}
                          }
                        }
                      }
                      None => {}
                    }
                    match matchable_pos_sets_2.get(&pos_pair_2.1) {
                      Some(matchable_poss) => {
                        for &x in matchable_poss {
                          if x <= pos_pair_2.0 {
                            continue;
                          }
                          let mut insert_prob = NEG_INFINITY;
                          let long_x = x.to_usize().unwrap();
                          let insert_score_range = insert_score_range_sets.insert_scores_range
                            [long_pos_pair_2.0][long_x - 1]
                            + feature_score_sets.match_2_insert_count;
                          match backward_tmp_part_func_set_mat.get(&(x, pos_pair_2.1)) {
                            Some(part_funcs) => {
                              let backward_term = part_funcs.part_func_on_sa;
                              let term =
                                prob_coeff + forward_term + insert_score_range + backward_term;
                              logsumexp(&mut insert_prob, term);
                              logsumexp(
                                &mut expected_feature_count_sets.match_2_insert_count,
                                (2. as Prob).ln() + insert_prob,
                              );
                              let pos_pair_4 = (pos_pair_2.0, x - T::one());
                              match upp_mat_pair_range.0.get_mut(&pos_pair_4) {
                                Some(upp) => {
                                  logsumexp(upp, insert_prob);
                                }
                                None => {
                                  upp_mat_pair_range.0.insert(pos_pair_4, insert_prob);
                                }
                              }
                              if pos_pair_2.0 < x - T::one() {
                                logsumexp(
                                  &mut expected_feature_count_sets.insert_extend_count,
                                  ((long_x - long_pos_pair_2.0 - 1) as Prob).ln() + insert_prob,
                                );
                              }
                            }
                            None => {}
                          }
                        }
                      }
                      None => {}
                    }
                  }
                  None => {}
                }
              }
            }
          }
        }
      }
    }
    if need_indel_info {
      for (pos_pair, &upp) in &upp_mat_pair_range.0 {
        for i in range_inclusive(pos_pair.0, pos_pair.1) {
          let long_i = i.to_usize().unwrap();
          let base = seq_pair.0[long_i];
          logsumexp(&mut expected_feature_count_sets.insert_counts[base], upp);
        }
      }
      for (pos_pair, &upp) in &upp_mat_pair_range.1 {
        for i in range_inclusive(pos_pair.0, pos_pair.1) {
          let long_i = i.to_usize().unwrap();
          let base = seq_pair.1[long_i];
          logsumexp(&mut expected_feature_count_sets.insert_counts[base], upp);
        }
      }
    }
    if produce_struct_profs {
      for (pos_pair, &upp_4_el) in &upp_mat_pair_4_el_range.0 {
        for i in range_inclusive(pos_pair.0, pos_pair.1) {
          let long_i = i.to_usize().unwrap();
          logsumexp(&mut sta_prob_mats.upp_mat_pair_4_el.0[long_i], upp_4_el);
        }
      }
      for (pos_pair, &upp_4_el) in &upp_mat_pair_4_el_range.1 {
        for i in range_inclusive(pos_pair.0, pos_pair.1) {
          let long_i = i.to_usize().unwrap();
          logsumexp(&mut sta_prob_mats.upp_mat_pair_4_el.1[long_i], upp_4_el);
        }
      }
      for (pos_pair, &upp_4_hl) in &upp_mat_pair_4_hl_range.0 {
        for i in range_inclusive(pos_pair.0, pos_pair.1) {
          let long_i = i.to_usize().unwrap();
          logsumexp(&mut sta_prob_mats.upp_mat_pair_4_hl.0[long_i], upp_4_hl);
        }
      }
      for (pos_pair, &upp_4_hl) in &upp_mat_pair_4_hl_range.1 {
        for i in range_inclusive(pos_pair.0, pos_pair.1) {
          let long_i = i.to_usize().unwrap();
          logsumexp(&mut sta_prob_mats.upp_mat_pair_4_hl.1[long_i], upp_4_hl);
        }
      }
      for (i, upp) in sta_prob_mats.upp_mat_pair_4_ml.0.iter_mut().enumerate() {
        let mut sum = sta_prob_mats.upp_mat_pair_4_hl.0[i];
        logsumexp(&mut sum, sta_prob_mats.upp_mat_pair_4_2l.0[i]);
        logsumexp(&mut sum, sta_prob_mats.upp_mat_pair_4_el.0[i]);
        logsumexp(&mut sum, sta_prob_mats.bpp_mat_pair_2.0[i]);
        *upp = 1. - expf(sum);
      }
      for (i, upp) in sta_prob_mats.upp_mat_pair_4_ml.1.iter_mut().enumerate() {
        let mut sum = sta_prob_mats.upp_mat_pair_4_hl.1[i];
        logsumexp(&mut sum, sta_prob_mats.upp_mat_pair_4_2l.1[i]);
        logsumexp(&mut sum, sta_prob_mats.upp_mat_pair_4_el.1[i]);
        logsumexp(&mut sum, sta_prob_mats.bpp_mat_pair_2.1[i]);
        *upp = 1. - expf(sum);
      }
      for upp in sta_prob_mats.upp_mat_pair_4_hl.0.iter_mut() {
        *upp = expf(*upp);
      }
      for upp in sta_prob_mats.upp_mat_pair_4_hl.1.iter_mut() {
        *upp = expf(*upp);
      }
      for upp in sta_prob_mats.upp_mat_pair_4_2l.0.iter_mut() {
        *upp = expf(*upp);
      }
      for upp in sta_prob_mats.upp_mat_pair_4_2l.1.iter_mut() {
        *upp = expf(*upp);
      }
      for upp in sta_prob_mats.upp_mat_pair_4_el.0.iter_mut() {
        *upp = expf(*upp);
      }
      for upp in sta_prob_mats.upp_mat_pair_4_el.1.iter_mut() {
        *upp = expf(*upp);
      }
      for bpp in sta_prob_mats.bpp_mat_pair_2.0.iter_mut() {
        *bpp = expf(*bpp);
      }
      for bpp in sta_prob_mats.bpp_mat_pair_2.1.iter_mut() {
        *bpp = expf(*bpp);
      }
    }
    if produce_align_probs {
      for loop_align_prob in sta_prob_mats.loop_align_prob_mat.values_mut() {
        *loop_align_prob = expf(*loop_align_prob);
      }
      for basepair_align_prob in sta_prob_mats.basepair_align_prob_mat.values_mut() {
        *basepair_align_prob = expf(*basepair_align_prob);
      }
    }
    if train_score_params {
      for count in expected_feature_count_sets
        .hairpin_loop_length_counts
        .iter_mut()
      {
        *count = expf(*count);
      }
      for count in expected_feature_count_sets
        .bulge_loop_length_counts
        .iter_mut()
      {
        *count = expf(*count);
      }
      for count in expected_feature_count_sets
        .interior_loop_length_counts
        .iter_mut()
      {
        *count = expf(*count);
      }
      for count in expected_feature_count_sets
        .interior_loop_length_counts_symm
        .iter_mut()
      {
        *count = expf(*count);
      }
      for count in expected_feature_count_sets
        .interior_loop_length_counts_asymm
        .iter_mut()
      {
        *count = expf(*count);
      }
      for count_3d_mat in expected_feature_count_sets.stack_count_mat.iter_mut() {
        for count_2d_mat in count_3d_mat.iter_mut() {
          for counts in count_2d_mat.iter_mut() {
            for count in counts.iter_mut() {
              *count = expf(*count);
            }
          }
        }
      }
      for count_3d_mat in expected_feature_count_sets
        .terminal_mismatch_count_mat
        .iter_mut()
      {
        for count_2d_mat in count_3d_mat.iter_mut() {
          for counts in count_2d_mat.iter_mut() {
            for count in counts.iter_mut() {
              *count = expf(*count);
            }
          }
        }
      }
      for count_2d_mat in expected_feature_count_sets.left_dangle_count_mat.iter_mut() {
        for counts in count_2d_mat.iter_mut() {
          for count in counts.iter_mut() {
            *count = expf(*count);
          }
        }
      }
      for count_2d_mat in expected_feature_count_sets
        .right_dangle_count_mat
        .iter_mut()
      {
        for counts in count_2d_mat.iter_mut() {
          for count in counts.iter_mut() {
            *count = expf(*count);
          }
        }
      }
      for counts in expected_feature_count_sets
        .interior_loop_length_count_mat_explicit
        .iter_mut()
      {
        for count in counts.iter_mut() {
          *count = expf(*count);
        }
      }
      for count in expected_feature_count_sets
        .bulge_loop_0x1_length_counts
        .iter_mut()
      {
        *count = expf(*count);
      }
      for counts in expected_feature_count_sets
        .interior_loop_1x1_length_count_mat
        .iter_mut()
      {
        for count in counts.iter_mut() {
          *count = expf(*count);
        }
      }
      for counts in expected_feature_count_sets.helix_end_count_mat.iter_mut() {
        for count in counts.iter_mut() {
          *count = expf(*count);
        }
      }
      for counts in expected_feature_count_sets.base_pair_count_mat.iter_mut() {
        for count in counts.iter_mut() {
          *count = expf(*count);
        }
      }
      expected_feature_count_sets.multi_loop_base_count =
        expf(expected_feature_count_sets.multi_loop_base_count);
      expected_feature_count_sets.multi_loop_basepairing_count =
        expf(expected_feature_count_sets.multi_loop_basepairing_count);
      expected_feature_count_sets.multi_loop_accessible_baseunpairing_count =
        expf(expected_feature_count_sets.multi_loop_accessible_baseunpairing_count);
      expected_feature_count_sets.external_loop_accessible_basepairing_count =
        expf(expected_feature_count_sets.external_loop_accessible_basepairing_count);
      expected_feature_count_sets.external_loop_accessible_baseunpairing_count =
        expf(expected_feature_count_sets.external_loop_accessible_baseunpairing_count);
      expected_feature_count_sets.match_2_match_count =
        expf(expected_feature_count_sets.match_2_match_count);
      expected_feature_count_sets.match_2_insert_count =
        expf(expected_feature_count_sets.match_2_insert_count);
      expected_feature_count_sets.insert_extend_count =
        expf(expected_feature_count_sets.insert_extend_count);
      expected_feature_count_sets.init_match_count =
        expf(expected_feature_count_sets.init_match_count);
      expected_feature_count_sets.init_insert_count =
        expf(expected_feature_count_sets.init_insert_count);
      for count in expected_feature_count_sets.insert_counts.iter_mut() {
        *count = expf(*count);
      }
      for counts in expected_feature_count_sets.align_count_mat.iter_mut() {
        for count in counts.iter_mut() {
          *count = expf(*count);
        }
      }
    }
  }
  sta_prob_mats
}

pub fn get_diff(x: usize, y: usize) -> usize {
  max(x, y) - min(x, y)
}

pub fn get_hl_fe_trained(
  feature_score_sets: &FeatureCountSets,
  seq: SeqSlice,
  pp_closing_loop: &(usize, usize),
) -> FreeEnergy {
  let hl_len = pp_closing_loop.1 - pp_closing_loop.0 - 1;
  feature_score_sets.hairpin_loop_length_counts_cumulative[hl_len]
    + get_junction_fe_single_trained(feature_score_sets, seq, pp_closing_loop)
}

pub fn get_consprob_twoloop_score(
  feature_score_sets: &FeatureCountSets,
  seq: SeqSlice,
  pp_closing_loop: &(usize, usize),
  accessible_pp: &(usize, usize),
) -> FeatureCount {
  let accessible_bp = (seq[accessible_pp.0], seq[accessible_pp.1]);
  let fe = if pp_closing_loop.0 + 1 == accessible_pp.0 && pp_closing_loop.1 - 1 == accessible_pp.1 {
    get_stack_fe_trained(feature_score_sets, seq, pp_closing_loop, accessible_pp)
  } else if pp_closing_loop.0 + 1 == accessible_pp.0 || pp_closing_loop.1 - 1 == accessible_pp.1 {
    get_bl_fe_trained(feature_score_sets, seq, pp_closing_loop, accessible_pp)
  } else {
    get_il_fe_trained(feature_score_sets, seq, pp_closing_loop, accessible_pp)
  };
  fe + feature_score_sets.base_pair_count_mat[accessible_bp.0][accessible_bp.1]
}

pub fn get_stack_fe_trained(
  feature_score_sets: &FeatureCountSets,
  seq: SeqSlice,
  pp_closing_loop: &(usize, usize),
  accessible_pp: &(usize, usize),
) -> FreeEnergy {
  let bp_closing_loop = (seq[pp_closing_loop.0], seq[pp_closing_loop.1]);
  let accessible_bp = (seq[accessible_pp.0], seq[accessible_pp.1]);
  feature_score_sets.stack_count_mat[bp_closing_loop.0][bp_closing_loop.1][accessible_bp.0]
    [accessible_bp.1]
}

pub fn get_bl_fe_trained(
  feature_score_sets: &FeatureCountSets,
  seq: SeqSlice,
  pp_closing_loop: &(usize, usize),
  accessible_pp: &(usize, usize),
) -> FreeEnergy {
  let bl_len = accessible_pp.0 - pp_closing_loop.0 + pp_closing_loop.1 - accessible_pp.1 - 2;
  let fe = if bl_len == 1 {
    feature_score_sets.bulge_loop_0x1_length_counts[if accessible_pp.0 - pp_closing_loop.0 - 1 == 1
    {
      seq[pp_closing_loop.0 + 1]
    } else {
      seq[pp_closing_loop.1 - 1]
    }]
  } else {
    0.
  };
  fe + feature_score_sets.bulge_loop_length_counts_cumulative[bl_len - 1]
    + get_junction_fe_single_trained(feature_score_sets, seq, pp_closing_loop)
    + get_junction_fe_single_trained(feature_score_sets, seq, &(accessible_pp.1, accessible_pp.0))
}

pub fn get_il_fe_trained(
  feature_score_sets: &FeatureCountSets,
  seq: SeqSlice,
  pp_closing_loop: &(usize, usize),
  accessible_pp: &(usize, usize),
) -> FreeEnergy {
  let pair_of_nums_of_unpaired_bases = (
    accessible_pp.0 - pp_closing_loop.0 - 1,
    pp_closing_loop.1 - accessible_pp.1 - 1,
  );
  let il_len = pair_of_nums_of_unpaired_bases.0 + pair_of_nums_of_unpaired_bases.1;
  let fe = if pair_of_nums_of_unpaired_bases.0 == pair_of_nums_of_unpaired_bases.1 {
    let fe_3 = if il_len == 2 {
      feature_score_sets.interior_loop_1x1_length_count_mat[seq[pp_closing_loop.0 + 1]]
        [seq[pp_closing_loop.1 - 1]]
    } else {
      0.
    };
    fe_3
      + feature_score_sets.interior_loop_length_counts_symm_cumulative
        [pair_of_nums_of_unpaired_bases.0 - 1]
  } else {
    feature_score_sets.interior_loop_length_counts_asymm_cumulative[get_abs_diff(
      pair_of_nums_of_unpaired_bases.0,
      pair_of_nums_of_unpaired_bases.1,
    ) - 1]
  };
  let fe_2 = if pair_of_nums_of_unpaired_bases.0 <= 4 && pair_of_nums_of_unpaired_bases.1 <= 4 {
    feature_score_sets.interior_loop_length_count_mat_explicit[pair_of_nums_of_unpaired_bases.0 - 1]
      [pair_of_nums_of_unpaired_bases.1 - 1]
  } else {
    0.
  };
  fe + fe_2
    + feature_score_sets.interior_loop_length_counts_cumulative[il_len - 2]
    + get_junction_fe_single_trained(feature_score_sets, seq, pp_closing_loop)
    + get_junction_fe_single_trained(feature_score_sets, seq, &(accessible_pp.1, accessible_pp.0))
}

pub fn get_junction_fe_single_trained(
  feature_score_sets: &FeatureCountSets,
  seq: SeqSlice,
  pp: &(usize, usize),
) -> FreeEnergy {
  let bp = (seq[pp.0], seq[pp.1]);
  get_helix_closing_fe_trained(feature_score_sets, &bp)
    + get_terminal_mismatch_fe_trained(feature_score_sets, &bp, &(seq[pp.0 + 1], seq[pp.1 - 1]))
}

pub fn get_helix_closing_fe_trained(
  feature_score_sets: &FeatureCountSets,
  bp: &BasePair,
) -> FreeEnergy {
  feature_score_sets.helix_end_count_mat[bp.0][bp.1]
}

pub fn get_terminal_mismatch_fe_trained(
  feature_score_sets: &FeatureCountSets,
  bp: &BasePair,
  mismatch_bp: &BasePair,
) -> FreeEnergy {
  feature_score_sets.terminal_mismatch_count_mat[bp.0][bp.1][mismatch_bp.0][mismatch_bp.1]
}

pub fn get_junction_fe_multi_trained(
  feature_score_sets: &FeatureCountSets,
  seq: SeqSlice,
  pp: &(usize, usize),
  seq_len: usize,
) -> FreeEnergy {
  let bp = (seq[pp.0], seq[pp.1]);
  let five_prime_end = 1;
  let three_prime_end = seq_len - 2;
  get_helix_closing_fe_trained(feature_score_sets, &bp)
    + if pp.0 < three_prime_end {
      feature_score_sets.left_dangle_count_mat[bp.0][bp.1][seq[pp.0 + 1]]
    } else {
      0.
    }
    + if pp.1 > five_prime_end {
      feature_score_sets.right_dangle_count_mat[bp.0][bp.1][seq[pp.1 - 1]]
    } else {
      0.
    }
}

pub fn get_dict_min_stack(
  base_pair_closing_loop: &BasePair,
  base_pair_accessible: &BasePair,
) -> (BasePair, BasePair) {
  let stack = (*base_pair_closing_loop, *base_pair_accessible);
  let inverse_stack = (
    (base_pair_accessible.1, base_pair_accessible.0),
    (base_pair_closing_loop.1, base_pair_closing_loop.0),
  );
  if stack < inverse_stack {
    stack
  } else {
    inverse_stack
  }
}

pub fn get_dict_min_basepair_align(
  base_pair: &BasePair,
  base_pair_2: &BasePair,
) -> (BasePair, BasePair) {
  let basepair_align = (*base_pair, *base_pair_2);
  let inverse_basepair_align = (*base_pair_2, *base_pair);
  if basepair_align < inverse_basepair_align {
    basepair_align
  } else {
    inverse_basepair_align
  }
}

pub fn get_dict_min_align(align: &BasePair) -> BasePair {
  let inverse_align = (align.1, align.0);
  if *align < inverse_align {
    *align
  } else {
    inverse_align
  }
}

pub fn get_dict_min_loop_len_pair(loop_len_pair: &(usize, usize)) -> (usize, usize) {
  let inverse_loop_len_pair = (loop_len_pair.1, loop_len_pair.0);
  if *loop_len_pair < inverse_loop_len_pair {
    *loop_len_pair
  } else {
    inverse_loop_len_pair
  }
}

pub fn get_dict_min_nuc_pair(nuc_pair: &(usize, usize)) -> (usize, usize) {
  get_dict_min_loop_len_pair(nuc_pair)
}

pub fn get_dict_min_mismatch_pair(mismatch_pair: &BasePair) -> BasePair {
  get_dict_min_align(mismatch_pair)
}

pub fn get_dict_min_base_pair(base_pair: &BasePair) -> BasePair {
  get_dict_min_align(base_pair)
}

pub fn get_num_of_multiloop_baseunpairing_nucs(
  pos_pair_closing_loop: &(usize, usize),
  pos_pairs_in_loop: &Vec<(usize, usize)>,
  seq: SeqSlice,
) -> usize {
  let mut count = 0;
  let mut prev_pos_pair = (0, 0);
  for pos_pair_in_loop in pos_pairs_in_loop {
    let start = if prev_pos_pair == (0, 0) {
      pos_pair_closing_loop.0 + 1
    } else {
      prev_pos_pair.1 + 1
    };
    for i in start..pos_pair_in_loop.0 {
      count += if seq[i] != PSEUDO_BASE { 1 } else { 0 };
    }
    prev_pos_pair = *pos_pair_in_loop;
  }
  for i in prev_pos_pair.1 + 1..pos_pair_closing_loop.1 {
    count += if seq[i] != PSEUDO_BASE { 1 } else { 0 };
  }
  count
}

pub fn get_num_of_externalloop_baseunpairing_nucs(
  pos_pairs_in_loop: &Vec<(usize, usize)>,
  seq: SeqSlice,
) -> usize {
  let mut count = 0;
  let mut prev_pos_pair = (0, 0);
  for pos_pair_in_loop in pos_pairs_in_loop {
    let start = if prev_pos_pair == (0, 0) {
      0
    } else {
      prev_pos_pair.1 + 1
    };
    for i in start..pos_pair_in_loop.0 {
      count += if seq[i] != PSEUDO_BASE { 1 } else { 0 };
    }
    prev_pos_pair = *pos_pair_in_loop;
  }
  for i in prev_pos_pair.1 + 1..seq.len() {
    count += if seq[i] != PSEUDO_BASE { 1 } else { 0 };
  }
  count
}

pub fn consprob_trained<T>(
  thread_pool: &mut Pool,
  fasta_records: &FastaRecords,
  min_bpp: Prob,
  min_align_prob: Prob,
  produce_struct_profs: bool,
  produce_align_probs: bool,
  train_type: TrainType,
) -> (ProbMatSets<T>, AlignProbMatSetsWithRnaIdPairs<T>)
where
  T: HashIndex,
{
  let trained = FeatureCountSets::load_trained_score_params();
  let mut align_feature_score_sets = AlignFeatureCountSets::new(0.);
  copy_feature_count_sets_align(&mut align_feature_score_sets, &trained);
  let ref ref_2_align_feature_score_sets = align_feature_score_sets;
  let mut struct_feature_score_sets = StructFeatureCountSets::new(0.);
  copy_feature_count_sets_struct(&mut struct_feature_score_sets, &trained);
  let ref ref_2_struct_feature_score_sets = struct_feature_score_sets;
  let feature_score_sets = if matches!(train_type, TrainType::TrainedTransfer) {
    trained
  } else if matches!(train_type, TrainType::TrainedRandomInit) {
    FeatureCountSets::load_trained_score_params_random_init()
  } else {
    let mut transferred = FeatureCountSets::new(0.);
    transferred.transfer();
    transferred
  };
  let num_of_fasta_records = fasta_records.len();
  let mut sparse_bpp_mats = vec![SparseProbMat::<T>::new(); num_of_fasta_records];
  let mut max_bp_spans = vec![T::zero(); num_of_fasta_records];
  let mut bp_score_param_set_seqs = vec![BpScoreParamSets::<T>::new(); num_of_fasta_records];
  let ref ref_2_feature_score_sets = feature_score_sets;
  thread_pool.scoped(|scope| {
    for (sparse_bpp_mat, max_bp_span, fasta_record, bp_score_param_sets) in multizip((
      sparse_bpp_mats.iter_mut(),
      max_bp_spans.iter_mut(),
      fasta_records.iter(),
      bp_score_param_set_seqs.iter_mut(),
    )) {
      let seq_len = fasta_record.seq.len();
      scope.execute(move || {
        let bpp_mat = mccaskill_algo(
          &fasta_record.seq[1..seq_len - 1],
          true,
          false,
          ref_2_struct_feature_score_sets,
        )
        .0;
        *sparse_bpp_mat = sparsify_bpp_mat::<T>(&bpp_mat, min_bpp);
        *max_bp_span = get_max_bp_span::<T>(sparse_bpp_mat);
        *bp_score_param_sets = BpScoreParamSets::<T>::set_curr_params(
          ref_2_feature_score_sets,
          &fasta_record.seq[..],
          sparse_bpp_mat,
        );
      });
    }
  });
  let mut prob_mats_with_rna_id_pairs = StaProbMatsWithRnaIdPairs::<T>::default();
  let mut align_prob_mats_with_rna_id_pairs = SparseProbMatsWithRnaIdPairs::<T>::default();
  for rna_id_1 in 0..num_of_fasta_records {
    for rna_id_2 in rna_id_1 + 1..num_of_fasta_records {
      let rna_id_pair = (rna_id_1, rna_id_2);
      prob_mats_with_rna_id_pairs.insert(rna_id_pair, StaProbMats::<T>::origin());
      align_prob_mats_with_rna_id_pairs.insert(rna_id_pair, SparseProbMat::<T>::default());
    }
  }
  thread_pool.scoped(|scope| {
    for (rna_id_pair, align_prob_mat) in align_prob_mats_with_rna_id_pairs.iter_mut() {
      let seq_pair = (
        &fasta_records[rna_id_pair.0].seq[..],
        &fasta_records[rna_id_pair.1].seq[..],
      );
      scope.execute(move || {
        *align_prob_mat = sparsify_align_prob_mat(
          &durbin_algo(&seq_pair, ref_2_align_feature_score_sets),
          min_align_prob,
        );
      });
    }
  });
  thread_pool.scoped(|scope| {
    for (rna_id_pair, prob_mats) in prob_mats_with_rna_id_pairs.iter_mut() {
      let seq_pair = (
        &fasta_records[rna_id_pair.0].seq[..],
        &fasta_records[rna_id_pair.1].seq[..],
      );
      let seq_len_pair = (
        T::from_usize(seq_pair.0.len()).unwrap(),
        T::from_usize(seq_pair.1.len()).unwrap(),
      );
      let max_bp_span_pair = (max_bp_spans[rna_id_pair.0], max_bp_spans[rna_id_pair.1]);
      let bpp_mat_pair = (
        &sparse_bpp_mats[rna_id_pair.0],
        &sparse_bpp_mats[rna_id_pair.1],
      );
      let bp_score_param_set_pair = (
        &bp_score_param_set_seqs[rna_id_pair.0],
        &bp_score_param_set_seqs[rna_id_pair.1],
      );
      let ref align_prob_mat = align_prob_mats_with_rna_id_pairs[rna_id_pair];
      let (
        forward_pos_pair_mat_set,
        backward_pos_pair_mat_set,
        _,
        pos_quadruple_mat_with_len_pairs,
        matchable_pos_sets_1,
        matchable_pos_sets_2,
      ) = get_sparse_pos_sets(&bpp_mat_pair, align_prob_mat, &seq_len_pair);
      let ref ref_2_feature_score_sets = feature_score_sets;
      scope.execute(move || {
        *prob_mats = io_algo_4_prob_mats::<T>(
          &seq_pair,
          ref_2_feature_score_sets,
          &max_bp_span_pair,
          align_prob_mat,
          produce_struct_profs,
          false,
          &mut FeatureCountSets::new(NEG_INFINITY),
          &forward_pos_pair_mat_set,
          &backward_pos_pair_mat_set,
          &pos_quadruple_mat_with_len_pairs,
          &bp_score_param_set_pair,
          produce_align_probs,
          &matchable_pos_sets_1,
          &matchable_pos_sets_2,
        )
        .0;
      });
    }
  });
  let ref ref_2_prob_mats_with_rna_id_pairs = prob_mats_with_rna_id_pairs;
  let mut prob_mat_sets = vec![PctStaProbMats::<T>::origin(); num_of_fasta_records];
  thread_pool.scoped(|scope| {
    for (rna_id, prob_mats) in prob_mat_sets.iter_mut().enumerate() {
      let seq_len = fasta_records[rna_id].seq.len();
      scope.execute(move || {
        *prob_mats = pct_of_prob_mats::<T>(
          ref_2_prob_mats_with_rna_id_pairs,
          rna_id,
          num_of_fasta_records,
          seq_len,
          produce_struct_profs,
        );
      });
    }
  });
  let mut align_prob_mat_sets_with_rna_id_pairs = AlignProbMatSetsWithRnaIdPairs::<T>::default();
  if produce_align_probs {
    for rna_id_1 in 0..num_of_fasta_records {
      for rna_id_2 in rna_id_1 + 1..num_of_fasta_records {
        let rna_id_pair = (rna_id_1, rna_id_2);
        let ref prob_mats = prob_mats_with_rna_id_pairs[&rna_id_pair];
        let mut align_prob_mats = AlignProbMats::<T>::new();
        align_prob_mats.loop_align_prob_mat = prob_mats.loop_align_prob_mat.clone();
        align_prob_mats.basepair_align_prob_mat = prob_mats.basepair_align_prob_mat.clone();
        align_prob_mats.align_prob_mat = align_prob_mats.loop_align_prob_mat.clone();
        for (pos_quadruple, &bpap) in &align_prob_mats.basepair_align_prob_mat {
          let pos_pair = (pos_quadruple.0, pos_quadruple.2);
          match align_prob_mats.align_prob_mat.get_mut(&pos_pair) {
            Some(bap) => {
              *bap += bpap;
            }
            None => {
              align_prob_mats.align_prob_mat.insert(pos_pair, bpap);
            }
          }
          let pos_pair = (pos_quadruple.1, pos_quadruple.3);
          match align_prob_mats.align_prob_mat.get_mut(&pos_pair) {
            Some(bap) => {
              *bap += bpap;
            }
            None => {
              align_prob_mats.align_prob_mat.insert(pos_pair, bpap);
            }
          }
        }
        align_prob_mat_sets_with_rna_id_pairs.insert(rna_id_pair, align_prob_mats);
      }
    }
  }
  (prob_mat_sets, align_prob_mat_sets_with_rna_id_pairs)
}

pub fn constrain<'a, T>(
  thread_pool: &mut Pool,
  train_data: &mut TrainData<T>,
  output_file_path: &Path,
  enable_random_init: bool,
  learning_tolerance: FeatureCount,
) where
  T: HashIndex,
{
  let mut feature_score_sets = FeatureCountSets::new(0.);
  if enable_random_init {
    feature_score_sets.rand_init();
  } else {
    feature_score_sets.transfer();
  }
  for train_datum in train_data.iter_mut() {
    train_datum.set_curr_params(&feature_score_sets);
  }
  let mut old_feature_score_sets = feature_score_sets.clone();
  let mut old_cost = INFINITY;
  let mut old_acc = NEG_INFINITY;
  let mut costs = Probs::new();
  let mut accs = costs.clone();
  let mut count = 0;
  let mut regularizers = Regularizers::from(vec![1.; feature_score_sets.get_len()]);
  let num_of_data = train_data.len() as FeatureCount;
  loop {
    let ref ref_2_feature_score_sets = feature_score_sets;
    thread_pool.scoped(|scope| {
      for train_datum in train_data.iter_mut() {
        train_datum.expected_feature_count_sets = FeatureCountSets::new(NEG_INFINITY);
        let seq_pair = (&train_datum.seq_pair.0[..], &train_datum.seq_pair.1[..]);
        let ref max_bp_span_pair = train_datum.max_bp_span_pair;
        let ref mut expected_feature_count_sets = train_datum.expected_feature_count_sets;
        let ref mut acc = train_datum.acc;
        let ref mut part_func = train_datum.part_func;
        let ref forward_pos_pair_mat_set = train_datum.forward_pos_pair_mat_set;
        let ref backward_pos_pair_mat_set = train_datum.backward_pos_pair_mat_set;
        let ref pos_quadruple_mat_with_len_pairs = train_datum.pos_quadruple_mat_with_len_pairs;
        let ref matchable_pos_sets_1 = train_datum.matchable_pos_sets_1;
        let ref matchable_pos_sets_2 = train_datum.matchable_pos_sets_2;
        let ref align_prob_mat = train_datum.align_prob_mat;
        let ref align = train_datum.align;
        let bp_score_param_set_pair = (
          &train_datum.bp_score_param_set_pair.0,
          &train_datum.bp_score_param_set_pair.1,
        );
        scope.execute(move || {
          let result = io_algo_4_prob_mats::<T>(
            &seq_pair,
            ref_2_feature_score_sets,
            max_bp_span_pair,
            align_prob_mat,
            false,
            true,
            expected_feature_count_sets,
            forward_pos_pair_mat_set,
            backward_pos_pair_mat_set,
            pos_quadruple_mat_with_len_pairs,
            &bp_score_param_set_pair,
            true,
            matchable_pos_sets_1,
            matchable_pos_sets_2,
          );
          *acc = get_sps_expected::<T>(&seq_pair, &align, &result.0);
          *part_func = result.1;
        });
      }
    });
    let acc =
      train_data.iter().map(|x| x.acc).sum::<FeatureCount>() / train_data.len() as FeatureCount;
    let acc_change = acc - old_acc;
    if acc_change <= learning_tolerance {
      feature_score_sets = old_feature_score_sets.clone();
      println!(
        "Accuracy change {} is <= learning tolerance {}; training finished",
        acc_change, learning_tolerance
      );
      break;
    }
    old_feature_score_sets = feature_score_sets.clone();
    feature_score_sets.update(&train_data, &mut regularizers);
    for train_datum in train_data.iter_mut() {
      train_datum.set_curr_params(&feature_score_sets);
    }
    let cost = feature_score_sets.get_cost(&train_data[..], &regularizers);
    let avg_cost_change = (cost - old_cost) / num_of_data;
    if avg_cost_change >= 0. {
      println!(
        "Average cost change {} is not negative; training finished",
        avg_cost_change
      );
      feature_score_sets = old_feature_score_sets.clone();
      break;
    }
    costs.push(cost);
    accs.push(acc);
    println!("Epoch {} finished (current cost = {}, current accuracy = {}, average cost change = {}, accuracy change = {})", count + 1, cost, acc, avg_cost_change, acc_change);
    count += 1;
    old_cost = cost;
    old_acc = acc;
  }
  write_feature_score_sets_trained(&feature_score_sets, enable_random_init);
  write_logs(&costs, &accs, output_file_path);
}

pub fn remove_gaps(seq: &Seq) -> Seq {
  seq
    .iter()
    .filter(|&&x| x != PSEUDO_BASE)
    .map(|&x| x)
    .collect()
}

pub fn convert_with_gaps<'a>(seq: &'a [u8]) -> Seq {
  let mut new_seq = Seq::new();
  for &c in seq {
    let new_base = convert_char(c);
    new_seq.push(new_base);
  }
  new_seq
}

pub fn convert_char(c: u8) -> Base {
  match c {
    SMALL_A | BIG_A => A,
    SMALL_C | BIG_C => C,
    SMALL_G | BIG_G => G,
    SMALL_U | BIG_U => U,
    _ => PSEUDO_BASE,
  }
}

pub fn get_mismatch_pair(
  seq: SeqSlice,
  pos_pair: &(usize, usize),
  is_closing: bool,
) -> (usize, usize) {
  let mut mismatch_pair = if is_closing {
    (seq[pos_pair.1], seq[pos_pair.0])
  } else {
    (PSEUDO_BASE, PSEUDO_BASE)
  };
  if is_closing {
    for i in pos_pair.0 + 1..pos_pair.1 {
      let align_char = seq[i];
      if align_char != PSEUDO_BASE {
        mismatch_pair.0 = align_char;
        break;
      }
    }
    for i in (pos_pair.0 + 1..pos_pair.1).rev() {
      let align_char = seq[i];
      if align_char != PSEUDO_BASE {
        mismatch_pair.1 = align_char;
        break;
      }
    }
  } else {
    for i in (0..pos_pair.0).rev() {
      let align_char = seq[i];
      if align_char != PSEUDO_BASE {
        mismatch_pair.0 = align_char;
        break;
      }
    }
    let seq_len = seq.len();
    for i in pos_pair.1 + 1..seq_len {
      let align_char = seq[i];
      if align_char != PSEUDO_BASE {
        mismatch_pair.1 = align_char;
        break;
      }
    }
  }
  mismatch_pair
}

pub fn get_hairpin_loop_length(seq: SeqSlice, pos_pair: &(usize, usize)) -> usize {
  let mut hairpin_loop_length = 0;
  for i in pos_pair.0 + 1..pos_pair.1 {
    let align_char = seq[i];
    if align_char == PSEUDO_BASE {
      continue;
    }
    hairpin_loop_length += 1;
  }
  hairpin_loop_length
}

pub fn get_2loop_length_pair(
  seq: SeqSlice,
  pos_pair_closing_loop: &(usize, usize),
  pos_pair_in_loop: &(usize, usize),
) -> (usize, usize) {
  let mut twoloop_length_pair = (0, 0);
  for i in pos_pair_closing_loop.0 + 1..pos_pair_in_loop.0 {
    let align_char = seq[i];
    if align_char == PSEUDO_BASE {
      continue;
    }
    twoloop_length_pair.0 += 1;
  }
  for i in pos_pair_in_loop.1 + 1..pos_pair_closing_loop.1 {
    let align_char = seq[i];
    if align_char == PSEUDO_BASE {
      continue;
    }
    twoloop_length_pair.1 += 1;
  }
  twoloop_length_pair
}

pub fn convert_vec_2_struct(
  feature_counts: &FeatureCounts,
  use_cumulative_feature_counts: bool,
) -> FeatureCountSets {
  let mut f = FeatureCountSets::new(0.);
  let mut offset = 0;
  let len = f.hairpin_loop_length_counts.len();
  for i in 0..len {
    let count = feature_counts[offset + i];
    if use_cumulative_feature_counts {
      f.hairpin_loop_length_counts_cumulative[i] = count;
    } else {
      f.hairpin_loop_length_counts[i] = count;
    }
  }
  offset += len;
  let len = f.bulge_loop_length_counts.len();
  for i in 0..len {
    let count = feature_counts[offset + i];
    if use_cumulative_feature_counts {
      f.bulge_loop_length_counts_cumulative[i] = count;
    } else {
      f.bulge_loop_length_counts[i] = count;
    }
  }
  offset += len;
  let len = f.interior_loop_length_counts.len();
  for i in 0..len {
    let count = feature_counts[offset + i];
    if use_cumulative_feature_counts {
      f.interior_loop_length_counts_cumulative[i] = count;
    } else {
      f.interior_loop_length_counts[i] = count;
    }
  }
  offset += len;
  let len = f.interior_loop_length_counts_symm.len();
  for i in 0..len {
    let count = feature_counts[offset + i];
    if use_cumulative_feature_counts {
      f.interior_loop_length_counts_symm_cumulative[i] = count;
    } else {
      f.interior_loop_length_counts_symm[i] = count;
    }
  }
  offset += len;
  let len = f.interior_loop_length_counts_asymm.len();
  for i in 0..len {
    let count = feature_counts[offset + i];
    if use_cumulative_feature_counts {
      f.interior_loop_length_counts_asymm_cumulative[i] = count;
    } else {
      f.interior_loop_length_counts_asymm[i] = count;
    }
  }
  offset += len;
  let len = f.stack_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      for k in 0..len {
        for l in 0..len {
          if !is_canonical(&(k, l)) {
            continue;
          }
          let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
          if ((i, j), (k, l)) != dict_min_stack {
            continue;
          }
          f.stack_count_mat[i][j][k][l] = feature_counts[offset];
          offset += 1;
        }
      }
    }
  }
  let len = f.terminal_mismatch_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      for k in 0..len {
        for l in 0..len {
          f.terminal_mismatch_count_mat[i][j][k][l] = feature_counts[offset];
          offset += 1;
        }
      }
    }
  }
  let len = f.left_dangle_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      for k in 0..len {
        f.left_dangle_count_mat[i][j][k] = feature_counts[offset];
        offset += 1;
      }
    }
  }
  let len = f.right_dangle_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      for k in 0..len {
        f.right_dangle_count_mat[i][j][k] = feature_counts[offset];
        offset += 1;
      }
    }
  }
  let len = f.helix_end_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      f.helix_end_count_mat[i][j] = feature_counts[offset];
      offset += 1;
    }
  }
  let len = f.base_pair_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      let dict_min_base_pair = get_dict_min_base_pair(&(i, j));
      if (i, j) != dict_min_base_pair {
        continue;
      }
      f.base_pair_count_mat[i][j] = feature_counts[offset];
      offset += 1;
    }
  }
  let len = f.interior_loop_length_count_mat_explicit.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_loop_len_pair = get_dict_min_loop_len_pair(&(i, j));
      if (i, j) != dict_min_loop_len_pair {
        continue;
      }
      f.interior_loop_length_count_mat_explicit[i][j] = feature_counts[offset];
      offset += 1;
    }
  }
  let len = f.bulge_loop_0x1_length_counts.len();
  for i in 0..len {
    f.bulge_loop_0x1_length_counts[i] = feature_counts[offset + i];
  }
  offset += len;
  let len = f.interior_loop_1x1_length_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_nuc_pair = get_dict_min_nuc_pair(&(i, j));
      if (i, j) != dict_min_nuc_pair {
        continue;
      }
      f.interior_loop_1x1_length_count_mat[i][j] = feature_counts[offset];
      offset += 1;
    }
  }
  f.multi_loop_base_count = feature_counts[offset];
  offset += 1;
  f.multi_loop_basepairing_count = feature_counts[offset];
  offset += 1;
  f.multi_loop_accessible_baseunpairing_count = feature_counts[offset];
  offset += 1;
  f.external_loop_accessible_basepairing_count = feature_counts[offset];
  offset += 1;
  f.external_loop_accessible_baseunpairing_count = feature_counts[offset];
  offset += 1;
  f.match_2_match_count = feature_counts[offset];
  offset += 1;
  f.match_2_insert_count = feature_counts[offset];
  offset += 1;
  f.init_match_count = feature_counts[offset];
  offset += 1;
  f.insert_extend_count = feature_counts[offset];
  offset += 1;
  f.init_insert_count = feature_counts[offset];
  offset += 1;
  let len = f.insert_counts.len();
  for i in 0..len {
    f.insert_counts[i] = feature_counts[offset + i];
  }
  offset += len;
  let len = f.align_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_align = get_dict_min_align(&(i, j));
      if (i, j) != dict_min_align {
        continue;
      }
      f.align_count_mat[i][j] = feature_counts[offset];
      offset += 1;
    }
  }
  assert!(offset == f.get_len());
  f
}

pub fn convert_struct_2_vec(
  feature_count_sets: &FeatureCountSets,
  use_cumulative_feature_counts: bool,
) -> FeatureCounts {
  let f = feature_count_sets;
  let mut feature_counts = vec![0.; f.get_len()];
  let mut offset = 0;
  let len = f.hairpin_loop_length_counts.len();
  for i in 0..len {
    feature_counts[offset + i] = if use_cumulative_feature_counts {
      f.hairpin_loop_length_counts_cumulative[i]
    } else {
      f.hairpin_loop_length_counts[i]
    };
  }
  offset += len;
  let len = f.bulge_loop_length_counts.len();
  for i in 0..len {
    feature_counts[offset + i] = if use_cumulative_feature_counts {
      f.bulge_loop_length_counts_cumulative[i]
    } else {
      f.bulge_loop_length_counts[i]
    };
  }
  offset += len;
  let len = f.interior_loop_length_counts.len();
  for i in 0..len {
    feature_counts[offset + i] = if use_cumulative_feature_counts {
      f.interior_loop_length_counts_cumulative[i]
    } else {
      f.interior_loop_length_counts[i]
    };
  }
  offset += len;
  let len = f.interior_loop_length_counts_symm.len();
  for i in 0..len {
    feature_counts[offset + i] = if use_cumulative_feature_counts {
      f.interior_loop_length_counts_symm_cumulative[i]
    } else {
      f.interior_loop_length_counts_symm[i]
    };
  }
  offset += len;
  let len = f.interior_loop_length_counts_asymm.len();
  for i in 0..len {
    feature_counts[offset + i] = if use_cumulative_feature_counts {
      f.interior_loop_length_counts_asymm_cumulative[i]
    } else {
      f.interior_loop_length_counts_asymm[i]
    };
  }
  offset += len;
  let len = f.stack_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      for k in 0..len {
        for l in 0..len {
          if !is_canonical(&(k, l)) {
            continue;
          }
          let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
          if ((i, j), (k, l)) != dict_min_stack {
            continue;
          }
          feature_counts[offset] = f.stack_count_mat[i][j][k][l];
          offset += 1;
        }
      }
    }
  }
  let len = f.terminal_mismatch_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      for k in 0..len {
        for l in 0..len {
          feature_counts[offset] = f.terminal_mismatch_count_mat[i][j][k][l];
          offset += 1;
        }
      }
    }
  }
  let len = f.left_dangle_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      for k in 0..len {
        feature_counts[offset] = f.left_dangle_count_mat[i][j][k];
        offset += 1;
      }
    }
  }
  let len = f.right_dangle_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      for k in 0..len {
        feature_counts[offset] = f.right_dangle_count_mat[i][j][k];
        offset += 1;
      }
    }
  }
  let len = f.helix_end_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      feature_counts[offset] = f.helix_end_count_mat[i][j];
      offset += 1;
    }
  }
  let len = f.base_pair_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      let dict_min_base_pair = get_dict_min_base_pair(&(i, j));
      if (i, j) != dict_min_base_pair {
        continue;
      }
      feature_counts[offset] = f.base_pair_count_mat[i][j];
      offset += 1;
    }
  }
  let len = f.interior_loop_length_count_mat_explicit.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_loop_len_pair = get_dict_min_loop_len_pair(&(i, j));
      if (i, j) != dict_min_loop_len_pair {
        continue;
      }
      feature_counts[offset] = f.interior_loop_length_count_mat_explicit[i][j];
      offset += 1;
    }
  }
  let len = f.bulge_loop_0x1_length_counts.len();
  for i in 0..len {
    feature_counts[offset + i] = f.bulge_loop_0x1_length_counts[i];
  }
  offset += len;
  let len = f.interior_loop_1x1_length_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_nuc_pair = get_dict_min_nuc_pair(&(i, j));
      if (i, j) != dict_min_nuc_pair {
        continue;
      }
      feature_counts[offset] = f.interior_loop_1x1_length_count_mat[i][j];
      offset += 1;
    }
  }
  feature_counts[offset] = f.multi_loop_base_count;
  offset += 1;
  feature_counts[offset] = f.multi_loop_basepairing_count;
  offset += 1;
  feature_counts[offset] = f.multi_loop_accessible_baseunpairing_count;
  offset += 1;
  feature_counts[offset] = f.external_loop_accessible_basepairing_count;
  offset += 1;
  feature_counts[offset] = f.external_loop_accessible_baseunpairing_count;
  offset += 1;
  feature_counts[offset] = f.match_2_match_count;
  offset += 1;
  feature_counts[offset] = f.match_2_insert_count;
  offset += 1;
  feature_counts[offset] = f.init_match_count;
  offset += 1;
  feature_counts[offset] = f.insert_extend_count;
  offset += 1;
  feature_counts[offset] = f.init_insert_count;
  offset += 1;
  let len = f.insert_counts.len();
  for i in 0..len {
    feature_counts[offset + i] = f.insert_counts[i];
  }
  offset += len;
  let len = f.align_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_align = get_dict_min_align(&(i, j));
      if (i, j) != dict_min_align {
        continue;
      }
      feature_counts[offset] = f.align_count_mat[i][j];
      offset += 1;
    }
  }
  assert!(offset == f.get_len());
  Array::from(feature_counts)
}

pub fn convert_feature_counts_2_bfgs_feature_counts(
  feature_counts: &FeatureCounts,
) -> BfgsFeatureCounts {
  let vec: Vec<BfgsFeatureCount> = feature_counts
    .to_vec()
    .iter()
    .map(|x| *x as BfgsFeatureCount)
    .collect();
  BfgsFeatureCounts::from(vec)
}

pub fn convert_bfgs_feature_counts_2_feature_counts(
  feature_counts: &BfgsFeatureCounts,
) -> FeatureCounts {
  let vec: Vec<FeatureCount> = feature_counts
    .to_vec()
    .iter()
    .map(|x| *x as FeatureCount)
    .collect();
  FeatureCounts::from(vec)
}

pub fn get_regularizer(group_size: usize, squared_sum: FeatureCount) -> Regularizer {
  (group_size as FeatureCount / 2. + GAMMA_DIST_ALPHA) / (squared_sum / 2. + GAMMA_DIST_BETA)
}

pub fn write_feature_score_sets_trained(
  feature_score_sets: &FeatureCountSets,
  enable_random_init: bool,
) {
  let mut writer_2_trained_feature_score_sets_file = BufWriter::new(
    File::create(if enable_random_init {
      TRAINED_FEATURE_SCORE_SETS_FILE_PATH_RANDOM_INIT
    } else {
      TRAINED_FEATURE_SCORE_SETS_FILE_PATH
    })
    .unwrap(),
  );
  let mut buf_4_writer_2_trained_feature_score_sets_file = format!("use FeatureCountSets;\nimpl FeatureCountSets {{\npub fn load_trained_score_params{}() -> FeatureCountSets {{\nFeatureCountSets {{\nhairpin_loop_length_counts: ", if enable_random_init {"_random_init"} else {""});
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nbulge_loop_length_counts: ",
    &feature_score_sets.hairpin_loop_length_counts
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\ninterior_loop_length_counts: ",
    &feature_score_sets.bulge_loop_length_counts
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\ninterior_loop_length_counts_symm: ",
    &feature_score_sets.interior_loop_length_counts
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\ninterior_loop_length_counts_asymm: ",
    &feature_score_sets.interior_loop_length_counts_symm
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nstack_count_mat: ",
    &feature_score_sets.interior_loop_length_counts_asymm
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nterminal_mismatch_count_mat: ",
    &feature_score_sets.stack_count_mat
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nleft_dangle_count_mat: ",
    &feature_score_sets.terminal_mismatch_count_mat
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nright_dangle_count_mat: ",
    &feature_score_sets.left_dangle_count_mat
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nhelix_end_count_mat: ",
    &feature_score_sets.right_dangle_count_mat
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nbase_pair_count_mat: ",
    &feature_score_sets.helix_end_count_mat
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\ninterior_loop_length_count_mat_explicit: ",
    &feature_score_sets.base_pair_count_mat
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nbulge_loop_0x1_length_counts: ",
    &feature_score_sets.interior_loop_length_count_mat_explicit
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\ninterior_loop_1x1_length_count_mat: ",
    &feature_score_sets.bulge_loop_0x1_length_counts
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nmulti_loop_base_count: ",
    &feature_score_sets.interior_loop_1x1_length_count_mat
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nmulti_loop_basepairing_count: ",
    feature_score_sets.multi_loop_base_count
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nmulti_loop_accessible_baseunpairing_count: ",
    feature_score_sets.multi_loop_basepairing_count
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nexternal_loop_accessible_basepairing_count: ",
    feature_score_sets.multi_loop_accessible_baseunpairing_count
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nexternal_loop_accessible_baseunpairing_count: ",
    feature_score_sets.external_loop_accessible_basepairing_count
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nmatch_2_match_count: ",
    feature_score_sets.external_loop_accessible_baseunpairing_count
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nmatch_2_insert_count: ",
    feature_score_sets.match_2_match_count
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\ninsert_extend_count: ",
    feature_score_sets.match_2_insert_count
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\ninit_match_count: ",
    feature_score_sets.insert_extend_count
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\ninit_insert_count: ",
    feature_score_sets.init_match_count
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\ninsert_counts: ",
    feature_score_sets.init_insert_count
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nalign_count_mat: ",
    feature_score_sets.insert_counts
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nhairpin_loop_length_counts_cumulative: ",
    feature_score_sets.align_count_mat
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\nbulge_loop_length_counts_cumulative: ",
    &feature_score_sets.hairpin_loop_length_counts_cumulative
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\ninterior_loop_length_counts_cumulative: ",
    &feature_score_sets.bulge_loop_length_counts_cumulative
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\ninterior_loop_length_counts_symm_cumulative: ",
    &feature_score_sets.interior_loop_length_counts_cumulative
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},\ninterior_loop_length_counts_asymm_cumulative: ",
    &feature_score_sets.interior_loop_length_counts_symm_cumulative
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&format!(
    "{:?},",
    &feature_score_sets.interior_loop_length_counts_asymm_cumulative
  ));
  buf_4_writer_2_trained_feature_score_sets_file.push_str(&String::from("\n}\n}\n}"));
  let _ = writer_2_trained_feature_score_sets_file
    .write_all(buf_4_writer_2_trained_feature_score_sets_file.as_bytes());
}

pub fn write_logs(costs: &Probs, accs: &Probs, output_file_path: &Path) {
  let mut writer_2_output_file = BufWriter::new(File::create(output_file_path).unwrap());
  let mut buf_4_writer_2_output_file = String::new();
  for (cost, acc) in costs.iter().zip(accs.iter()) {
    buf_4_writer_2_output_file.push_str(&format!("{},{}\n", cost, acc));
  }
  let _ = writer_2_output_file.write_all(buf_4_writer_2_output_file.as_bytes());
}

pub fn copy_feature_count_sets_align(
  align_feature_count_sets: &mut AlignFeatureCountSets,
  feature_count_sets: &FeatureCountSets,
) {
  align_feature_count_sets.match_2_match_count = feature_count_sets.match_2_match_count;
  align_feature_count_sets.match_2_insert_count = feature_count_sets.match_2_insert_count;
  align_feature_count_sets.insert_extend_count = feature_count_sets.insert_extend_count;
  align_feature_count_sets.init_match_count = feature_count_sets.init_match_count;
  align_feature_count_sets.init_insert_count = feature_count_sets.init_insert_count;
  let len = align_feature_count_sets.insert_counts.len();
  for i in 0..len {
    align_feature_count_sets.insert_counts[i] = feature_count_sets.insert_counts[i];
  }
  let len = align_feature_count_sets.align_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      align_feature_count_sets.align_count_mat[i][j] = feature_count_sets.align_count_mat[i][j];
    }
  }
}

pub fn copy_feature_count_sets_struct(
  struct_feature_count_sets: &mut StructFeatureCountSets,
  feature_count_sets: &FeatureCountSets,
) {
  for (v, &w) in struct_feature_count_sets
    .hairpin_loop_length_counts
    .iter_mut()
    .zip(feature_count_sets.hairpin_loop_length_counts.iter())
  {
    *v = w;
  }
  for (v, &w) in struct_feature_count_sets
    .bulge_loop_length_counts
    .iter_mut()
    .zip(feature_count_sets.bulge_loop_length_counts.iter())
  {
    *v = w;
  }
  for (v, &w) in struct_feature_count_sets
    .interior_loop_length_counts
    .iter_mut()
    .zip(feature_count_sets.interior_loop_length_counts.iter())
  {
    *v = w;
  }
  for (v, &w) in struct_feature_count_sets
    .interior_loop_length_counts_symm
    .iter_mut()
    .zip(feature_count_sets.interior_loop_length_counts_symm.iter())
  {
    *v = w;
  }
  for (v, &w) in struct_feature_count_sets
    .interior_loop_length_counts_asymm
    .iter_mut()
    .zip(feature_count_sets.interior_loop_length_counts_asymm.iter())
  {
    *v = w;
  }
  let len = struct_feature_count_sets.stack_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      for k in 0..len {
        for l in 0..len {
          if !is_canonical(&(k, l)) {
            continue;
          }
          struct_feature_count_sets.stack_count_mat[i][j][k][l] =
            feature_count_sets.stack_count_mat[i][j][k][l];
        }
      }
    }
  }
  let len = struct_feature_count_sets.terminal_mismatch_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      for k in 0..len {
        for l in 0..len {
          struct_feature_count_sets.terminal_mismatch_count_mat[i][j][k][l] =
            feature_count_sets.terminal_mismatch_count_mat[i][j][k][l];
        }
      }
    }
  }
  let len = struct_feature_count_sets.left_dangle_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      for k in 0..len {
        struct_feature_count_sets.left_dangle_count_mat[i][j][k] =
          feature_count_sets.left_dangle_count_mat[i][j][k];
      }
    }
  }
  let len = struct_feature_count_sets.right_dangle_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      for k in 0..len {
        struct_feature_count_sets.right_dangle_count_mat[i][j][k] =
          feature_count_sets.right_dangle_count_mat[i][j][k];
      }
    }
  }
  let len = struct_feature_count_sets.helix_end_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      struct_feature_count_sets.helix_end_count_mat[i][j] =
        feature_count_sets.helix_end_count_mat[i][j];
    }
  }
  let len = struct_feature_count_sets.base_pair_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      struct_feature_count_sets.base_pair_count_mat[i][j] =
        feature_count_sets.base_pair_count_mat[i][j];
    }
  }
  let len = struct_feature_count_sets
    .interior_loop_length_count_mat_explicit
    .len();
  for i in 0..len {
    for j in 0..len {
      struct_feature_count_sets.interior_loop_length_count_mat_explicit[i][j] =
        feature_count_sets.interior_loop_length_count_mat_explicit[i][j];
    }
  }
  for (v, &w) in struct_feature_count_sets
    .bulge_loop_0x1_length_counts
    .iter_mut()
    .zip(feature_count_sets.bulge_loop_0x1_length_counts.iter())
  {
    *v = w;
  }
  let len = struct_feature_count_sets
    .interior_loop_1x1_length_count_mat
    .len();
  for i in 0..len {
    for j in 0..len {
      struct_feature_count_sets.interior_loop_1x1_length_count_mat[i][j] =
        feature_count_sets.interior_loop_1x1_length_count_mat[i][j];
    }
  }
  struct_feature_count_sets.multi_loop_base_count = feature_count_sets.multi_loop_base_count;
  struct_feature_count_sets.multi_loop_basepairing_count =
    feature_count_sets.multi_loop_basepairing_count;
  struct_feature_count_sets.multi_loop_accessible_baseunpairing_count =
    feature_count_sets.multi_loop_accessible_baseunpairing_count;
  struct_feature_count_sets.external_loop_accessible_basepairing_count =
    feature_count_sets.external_loop_accessible_basepairing_count;
  struct_feature_count_sets.external_loop_accessible_baseunpairing_count =
    feature_count_sets.external_loop_accessible_baseunpairing_count;
  struct_feature_count_sets.accumulate();
}

pub fn print_train_info(feature_score_sets: &FeatureCountSets) {
  let mut num_of_groups = 0;
  println!("Training the parameter groups below");
  println!("-----------------------------------");
  println!("Groups from the CONTRAfold model...");
  println!(
    "Hairpin loop length (group size {})",
    feature_score_sets.hairpin_loop_length_counts.len()
  );
  num_of_groups += 1;
  println!(
    "Bulge loop length (group size {})",
    feature_score_sets.bulge_loop_length_counts.len()
  );
  num_of_groups += 1;
  println!(
    "Interior loop length (group size {})",
    feature_score_sets.interior_loop_length_counts.len()
  );
  num_of_groups += 1;
  println!(
    "Interior loop length symmetric (group size {})",
    feature_score_sets.interior_loop_length_counts_symm.len()
  );
  num_of_groups += 1;
  println!(
    "Interior loop length asymmetric (group size {})",
    feature_score_sets.interior_loop_length_counts_asymm.len()
  );
  num_of_groups += 1;
  let mut group_size = 0;
  let len = feature_score_sets.stack_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      for k in 0..len {
        for l in 0..len {
          if !is_canonical(&(k, l)) {
            continue;
          }
          let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
          if ((i, j), (k, l)) != dict_min_stack {
            continue;
          }
          group_size += 1;
        }
      }
    }
  }
  println!("Stacking (group size {})", group_size);
  num_of_groups += 1;
  let mut group_size = 0;
  let len = feature_score_sets.terminal_mismatch_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      for _ in 0..len {
        for _ in 0..len {
          group_size += 1;
        }
      }
    }
  }
  println!("Terminal mismatch (group size {})", group_size);
  num_of_groups += 1;
  let mut group_size = 0;
  let len = feature_score_sets.left_dangle_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      for _ in 0..len {
        group_size += 1;
      }
    }
  }
  let len = feature_score_sets.right_dangle_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      for _ in 0..len {
        group_size += 1;
      }
    }
  }
  println!("Dangling (group size {})", group_size);
  num_of_groups += 1;
  let mut group_size = 0;
  let len = feature_score_sets.helix_end_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      group_size += 1;
    }
  }
  println!("Helix end (group size {})", group_size);
  num_of_groups += 1;
  let mut group_size = 0;
  let len = feature_score_sets.base_pair_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      if !is_canonical(&(i, j)) {
        continue;
      }
      let dict_min_base_pair = get_dict_min_base_pair(&(i, j));
      if (i, j) != dict_min_base_pair {
        continue;
      }
      group_size += 1;
    }
  }
  println!("Base-pairing (group size {})", group_size);
  num_of_groups += 1;
  let mut group_size = 0;
  let len = feature_score_sets
    .interior_loop_length_count_mat_explicit
    .len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_loop_len_pair = get_dict_min_loop_len_pair(&(i, j));
      if (i, j) != dict_min_loop_len_pair {
        continue;
      }
      group_size += 1;
    }
  }
  println!("Interior loop length explicit (group size {})", group_size);
  num_of_groups += 1;
  println!(
    "Bulge loop length 0x1 (group size {})",
    feature_score_sets.bulge_loop_0x1_length_counts.len()
  );
  num_of_groups += 1;
  let mut group_size = 0;
  let len = feature_score_sets.interior_loop_1x1_length_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_nuc_pair = get_dict_min_nuc_pair(&(i, j));
      if (i, j) != dict_min_nuc_pair {
        continue;
      }
      group_size += 1;
    }
  }
  println!("Interior loop length 1x1 (group size {})", group_size);
  num_of_groups += 1;
  println!(
    "Multi-loop length (group size {})",
    CONSPROB_MULTI_LOOP_GROUP_SIZE
  );
  num_of_groups += 1;
  println!(
    "External-loop length (group size {})",
    CONSPROB_EXTERNAL_LOOP_GROUP_SIZE
  );
  num_of_groups += 1;
  println!("-----------------------------------");
  println!("Groups from the CONTRAlign model...");
  println!(
    "Match transition (group size {})",
    CONSPROB_MATCH_TRANSITION_GROUP_SIZE
  );
  num_of_groups += 1;
  println!(
    "Insert transition (group size {})",
    CONSPROB_INSERT_TRANSITION_GROUP_SIZE
  );
  num_of_groups += 1;
  println!(
    "Insert emission (group size {})",
    feature_score_sets.insert_counts.len()
  );
  num_of_groups += 1;
  let mut group_size = 0;
  let len = feature_score_sets.align_count_mat.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_align = get_dict_min_align(&(i, j));
      if (i, j) != dict_min_align {
        continue;
      }
      group_size += 1;
    }
  }
  println!("Match emission (group size {})", group_size);
  num_of_groups += 1;
  println!("-----------------------------------");
  println!(
    "Total # scoring parameters (from {} groups) to be trained: {}",
    num_of_groups,
    feature_score_sets.get_len()
  );
}
