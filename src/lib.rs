extern crate bio;
extern crate consprob;
extern crate my_bfgs as bfgs;
extern crate ndarray_rand;
extern crate rand;
pub mod trained_alignfold_scores;
pub mod trained_alignfold_scores_randinit;

pub use bfgs::bfgs;
pub use bio::io::fasta::*;
pub use bio::utils::*;
pub use consprob::*;
pub use ndarray_rand::rand_distr::{Distribution, Normal};
pub use ndarray_rand::RandomExt;
pub use rand::thread_rng;
pub use std::f32::INFINITY;
pub use std::fs::{read_dir, DirEntry};
pub use std::io::stdout;

pub type SparsePosMat<T> = HashSet<PosPair<T>>;
pub type FoldScoresPairTrained<T> = (FoldScoresTrained<T>, FoldScoresTrained<T>);
pub type RefFoldScoresPair<'a, T> = (&'a FoldScoresTrained<T>, &'a FoldScoresTrained<T>);
pub type Regularizers = Array1<Regularizer>;
pub type Regularizer = Score;
pub type BfgsScores = Array1<BfgsScore>;
pub type BfgsScore = f64;
pub type Scores = Array1<Score>;
pub type TrainData<T> = Vec<TrainDatum<T>>;
pub type RealSeqPair = (Seq, Seq);
pub type LoopStruct = HashMap<(usize, usize), Vec<(usize, usize)>>;

#[derive(Clone)]
pub struct FoldScoresTrained<T> {
  pub hairpin_scores: SparseScoreMat<T>,
  pub twoloop_scores: ScoreMat4d<T>,
  pub multibranch_close_scores: SparseScoreMat<T>,
  pub multibranch_accessible_scores: SparseScoreMat<T>,
  pub external_accessible_scores: SparseScoreMat<T>,
}

#[derive(Clone)]
pub struct TrainDatum<T> {
  pub seq_pair: RealSeqPair,
  pub seq_pair_gapped: RealSeqPair,
  pub alignfold_counts_observed: AlignfoldScores,
  pub alignfold_counts_expected: AlignfoldScores,
  pub basepair_probs_pair: SparseProbMatPair<T>,
  pub max_basepair_span_pair: (T, T),
  pub global_sum: Prob,
  pub forward_pos_pairs: PosPairMatSet<T>,
  pub backward_pos_pairs: PosPairMatSet<T>,
  pub pos_quads_hashed_lens: PosQuadsHashedLens<T>,
  pub matchable_poss: SparsePosSets<T>,
  pub matchable_poss2: SparsePosSets<T>,
  pub fold_scores_pair: FoldScoresPairTrained<T>,
  pub match_probs: SparseProbMat<T>,
  pub alignfold: PairAlignfold<T>,
  pub accuracy: Score,
}

#[derive(Clone)]
pub struct PairAlignfold<T> {
  pub matched_pos_pairs: SparsePosMat<T>,
  pub inserted_poss: SparsePoss<T>,
  pub deleted_poss: SparsePoss<T>,
}

#[derive(Clone, Debug)]
pub struct AlignfoldScores {
  // The CONTRAfold model.
  pub hairpin_scores_len: HairpinScoresLen,
  pub bulge_scores_len: BulgeScoresLen,
  pub interior_scores_len: InteriorScoresLen,
  pub interior_scores_symmetric: InteriorScoresSymmetric,
  pub interior_scores_asymmetric: InteriorScoresAsymmetric,
  pub stack_scores: StackScores,
  pub terminal_mismatch_scores: TerminalMismatchScores,
  pub dangling_scores_left: DanglingScores,
  pub dangling_scores_right: DanglingScores,
  pub helix_close_scores: HelixCloseScores,
  pub basepair_scores: BasepairScores,
  pub interior_scores_explicit: InteriorScoresExplicit,
  pub bulge_scores_0x1: BulgeScores0x1,
  pub interior_scores_1x1: InteriorScores1x1Contra,
  pub multibranch_score_base: Score,
  pub multibranch_score_basepair: Score,
  pub multibranch_score_unpair: Score,
  pub external_score_basepair: Score,
  pub external_score_unpair: Score,
  // The CONTRAlign model.
  pub match2match_score: Score,
  pub match2insert_score: Score,
  pub insert_extend_score: Score,
  pub init_match_score: Score,
  pub init_insert_score: Score,
  pub insert_scores: InsertScores,
  pub match_scores: MatchScores,
  // The cumulative parameters of the CONTRAfold model.
  pub hairpin_scores_len_cumulative: HairpinScoresLen,
  pub bulge_scores_len_cumulative: BulgeScoresLen,
  pub interior_scores_len_cumulative: InteriorScoresLen,
  pub interior_scores_symmetric_cumulative: InteriorScoresSymmetric,
  pub interior_scores_asymmetric_cumulative: InteriorScoresAsymmetric,
}
pub type ScoreMat = Vec<Vec<Score>>;
pub struct RangeInsertScores {
  pub insert_scores: ScoreMat,
  pub insert_scores_external: ScoreMat,
  pub insert_scores_multibranch: ScoreMat,
  pub insert_scores2: ScoreMat,
  pub insert_scores_external2: ScoreMat,
  pub insert_scores_multibranch2: ScoreMat,
}

pub type InputsAlignfoldProbsGetter<'a, T> = (
  &'a SeqPair<'a>,
  &'a AlignfoldScores,
  &'a PosPair<T>,
  &'a SparseProbMat<T>,
  &'a AlignfoldSums<T>,
  bool,
  Sum,
  bool,
  &'a mut AlignfoldScores,
  &'a PosQuadsHashedLens<T>,
  &'a RefFoldScoresPair<'a, T>,
  bool,
  &'a PosPairMatSet<T>,
  &'a PosPairMatSet<T>,
  &'a RangeInsertScores,
  &'a SparsePosSets<T>,
  &'a SparsePosSets<T>,
);

pub type InputsConsprobCore<'a, T> = (
  &'a SeqPair<'a>,
  &'a AlignfoldScores,
  &'a PosPair<T>,
  &'a SparseProbMat<T>,
  bool,
  bool,
  &'a mut AlignfoldScores,
  &'a PosPairMatSet<T>,
  &'a PosPairMatSet<T>,
  &'a PosQuadsHashedLens<T>,
  &'a RefFoldScoresPair<'a, T>,
  bool,
  &'a SparsePosSets<T>,
  &'a SparsePosSets<T>,
);

pub type Inputs2loopSumsGetter<'a, T> = (
  &'a SeqPair<'a>,
  &'a AlignfoldScores,
  &'a SparseProbMat<T>,
  &'a PosQuad<T>,
  &'a AlignfoldSums<T>,
  bool,
  &'a PosPairMatSet<T>,
  &'a RefFoldScoresPair<'a, T>,
  &'a RangeInsertScores,
  &'a SparsePosSets<T>,
  &'a SparsePosSets<T>,
);

pub type InputsLoopSumsGetter<'a, T> = (
  &'a SeqPair<'a>,
  &'a AlignfoldScores,
  &'a SparseProbMat<T>,
  &'a PosQuad<T>,
  &'a mut AlignfoldSums<T>,
  bool,
  &'a PosPairMatSet<T>,
  &'a RangeInsertScores,
  &'a SparsePosSets<T>,
  &'a SparsePosSets<T>,
);

pub type InputsInsideSumsGetter<'a, T> = (
  &'a SeqPair<'a>,
  &'a AlignfoldScores,
  &'a PosPair<T>,
  &'a SparseProbMat<T>,
  bool,
  &'a PosPairMatSet<T>,
  &'a PosPairMatSet<T>,
  &'a PosQuadsHashedLens<T>,
  &'a RefFoldScoresPair<'a, T>,
  &'a RangeInsertScores,
  &'a SparsePosSets<T>,
  &'a SparsePosSets<T>,
);

impl<T: HashIndex> Default for FoldScoresTrained<T> {
  fn default() -> Self {
    Self::new()
  }
}

impl<T: HashIndex> FoldScoresTrained<T> {
  pub fn new() -> FoldScoresTrained<T> {
    FoldScoresTrained {
      hairpin_scores: SparseScoreMat::<T>::default(),
      twoloop_scores: ScoreMat4d::<T>::default(),
      multibranch_close_scores: SparseScoreMat::<T>::default(),
      multibranch_accessible_scores: SparseScoreMat::<T>::default(),
      external_accessible_scores: SparseScoreMat::<T>::default(),
    }
  }

  pub fn set_curr_scores(
    alignfold_scores: &AlignfoldScores,
    seq: SeqSlice,
    basepair_probs: &SparseProbMat<T>,
  ) -> FoldScoresTrained<T> {
    let mut fold_scores = FoldScoresTrained::<T>::new();
    for pos_pair in basepair_probs.keys() {
      let long_pos_pair = (
        pos_pair.0.to_usize().unwrap(),
        pos_pair.1.to_usize().unwrap(),
      );
      if long_pos_pair.1 - long_pos_pair.0 - 1 <= MAX_LOOP_LEN {
        let x = get_hairpin_score(alignfold_scores, seq, &long_pos_pair);
        fold_scores.hairpin_scores.insert(*pos_pair, x);
      }
      let multibranch_close_score = alignfold_scores.multibranch_score_base
        + alignfold_scores.multibranch_score_basepair
        + get_junction_score(alignfold_scores, seq, &long_pos_pair);
      fold_scores
        .multibranch_close_scores
        .insert(*pos_pair, multibranch_close_score);
      let basepair = (seq[long_pos_pair.0], seq[long_pos_pair.1]);
      let junction_score =
        get_junction_score(alignfold_scores, seq, &(long_pos_pair.1, long_pos_pair.0))
          + alignfold_scores.basepair_scores[basepair.0][basepair.1];
      let multibranch_accessible_score =
        junction_score + alignfold_scores.multibranch_score_basepair;
      fold_scores
        .multibranch_accessible_scores
        .insert(*pos_pair, multibranch_accessible_score);
      let external_accessible_score = junction_score + alignfold_scores.external_score_basepair;
      fold_scores
        .external_accessible_scores
        .insert(*pos_pair, external_accessible_score);
      for x in basepair_probs.keys() {
        if !(x.0 < pos_pair.0 && pos_pair.1 < x.1) {
          continue;
        }
        let y = (x.0.to_usize().unwrap(), x.1.to_usize().unwrap());
        if long_pos_pair.0 - y.0 - 1 + y.1 - long_pos_pair.1 - 1 > MAX_LOOP_LEN {
          continue;
        }
        let y = get_twoloop_score(alignfold_scores, seq, &y, &long_pos_pair);
        fold_scores
          .twoloop_scores
          .insert((x.0, x.1, pos_pair.0, pos_pair.1), y);
      }
    }
    fold_scores
  }
}

impl AlignfoldScores {
  pub fn new(init_val: Score) -> AlignfoldScores {
    let init_vals = [init_val; NUM_BASES];
    let mat_2d = [init_vals; NUM_BASES];
    let mat_3d = [mat_2d; NUM_BASES];
    let mat_4d = [mat_3d; NUM_BASES];
    AlignfoldScores {
      // The CONTRAfold model.
      hairpin_scores_len: [init_val; MAX_LOOP_LEN + 1],
      bulge_scores_len: [init_val; MAX_LOOP_LEN],
      interior_scores_len: [init_val; MAX_LOOP_LEN - 1],
      interior_scores_symmetric: [init_val; MAX_INTERIOR_SYMMETRIC],
      interior_scores_asymmetric: [init_val; MAX_INTERIOR_ASYMMETRIC],
      stack_scores: mat_4d,
      terminal_mismatch_scores: mat_4d,
      dangling_scores_left: mat_3d,
      dangling_scores_right: mat_3d,
      helix_close_scores: mat_2d,
      basepair_scores: mat_2d,
      interior_scores_explicit: [[init_val; MAX_INTERIOR_EXPLICIT]; MAX_INTERIOR_EXPLICIT],
      bulge_scores_0x1: init_vals,
      interior_scores_1x1: mat_2d,
      multibranch_score_base: init_val,
      multibranch_score_basepair: init_val,
      multibranch_score_unpair: init_val,
      external_score_basepair: init_val,
      external_score_unpair: init_val,
      // The CONTRAlign model.
      match2match_score: init_val,
      match2insert_score: init_val,
      init_match_score: init_val,
      insert_extend_score: init_val,
      init_insert_score: init_val,
      insert_scores: init_vals,
      match_scores: mat_2d,
      // The cumulative parameters of the CONTRAfold model.
      hairpin_scores_len_cumulative: [init_val; MAX_LOOP_LEN + 1],
      bulge_scores_len_cumulative: [init_val; MAX_LOOP_LEN],
      interior_scores_len_cumulative: [init_val; MAX_LOOP_LEN - 1],
      interior_scores_symmetric_cumulative: [init_val; MAX_INTERIOR_SYMMETRIC],
      interior_scores_asymmetric_cumulative: [init_val; MAX_INTERIOR_ASYMMETRIC],
    }
  }

  pub fn len(&self) -> usize {
    let mut sum = 0;
    sum += self.hairpin_scores_len.len();
    sum += self.bulge_scores_len.len();
    sum += self.interior_scores_len.len();
    sum += self.interior_scores_symmetric.len();
    sum += self.interior_scores_asymmetric.len();
    let len = self.stack_scores.len();
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for k in 0..len {
          for l in 0..len {
            if !has_canonical_basepair(&(k, l)) {
              continue;
            }
            let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
            if ((i, j), (k, l)) != dict_min_stack {
              continue;
            }
            sum += 1;
          }
        }
      }
    }
    let len = self.terminal_mismatch_scores.len();
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for _ in 0..len {
          for _ in 0..len {
            sum += 1;
          }
        }
      }
    }
    let len = self.dangling_scores_left.len();
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for _ in 0..len {
          sum += 1;
        }
      }
    }
    let len = self.dangling_scores_right.len();
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for _ in 0..len {
          sum += 1;
        }
      }
    }
    let len = self.helix_close_scores.len();
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        sum += 1;
      }
    }
    let len = self.basepair_scores.len();
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        let dict_min_basepair = get_dict_min_pair(&(i, j));
        if (i, j) != dict_min_basepair {
          continue;
        }
        sum += 1;
      }
    }
    let len = self.interior_scores_explicit.len();
    for i in 0..len {
      for j in 0..len {
        let dict_min_len_pair = get_dict_min_pair(&(i, j));
        if (i, j) != dict_min_len_pair {
          continue;
        }
        sum += 1;
      }
    }
    sum += self.bulge_scores_0x1.len();
    let len = self.interior_scores_1x1.len();
    for i in 0..len {
      for j in 0..len {
        let dict_min_basepair = get_dict_min_pair(&(i, j));
        if (i, j) != dict_min_basepair {
          continue;
        }
        sum += 1;
      }
    }
    sum += GROUP_SIZE_MULTIBRANCH;
    sum += GROUP_SIZE_EXTERNAL;
    sum += GROUP_SIZE_MATCH_TRANSITION;
    sum += GROUP_SIZE_INSERT_TRANSITION;
    sum += self.insert_scores.len();
    let len = self.match_scores.len();
    for i in 0..len {
      for j in 0..len {
        let dict_min_match = get_dict_min_pair(&(i, j));
        if (i, j) != dict_min_match {
          continue;
        }
        sum += 1;
      }
    }
    sum
  }

  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  pub fn update_regularizers(&self, regularizers: &mut Regularizers) {
    let mut regularizers2 = vec![0.; regularizers.len()];
    let mut offset = 0;
    let len = self.hairpin_scores_len.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0..len {
      let x = self.hairpin_scores_len[i];
      squared_sum += x * x;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0..len {
      regularizers2[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.bulge_scores_len.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0..len {
      let x = self.bulge_scores_len[i];
      squared_sum += x * x;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0..len {
      regularizers2[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.interior_scores_len.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0..len {
      let x = self.interior_scores_len[i];
      squared_sum += x * x;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0..len {
      regularizers2[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.interior_scores_symmetric.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0..len {
      let x = self.interior_scores_symmetric[i];
      squared_sum += x * x;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0..len {
      regularizers2[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.interior_scores_asymmetric.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0..len {
      let x = self.interior_scores_asymmetric[i];
      squared_sum += x * x;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0..len {
      regularizers2[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.stack_scores.len();
    let mut effective_group_size = 0;
    let mut squared_sum = 0.;
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for k in 0..len {
          for l in 0..len {
            if !has_canonical_basepair(&(k, l)) {
              continue;
            }
            let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
            if ((i, j), (k, l)) != dict_min_stack {
              continue;
            }
            let x = self.stack_scores[i][j][k][l];
            squared_sum += x * x;
            effective_group_size += 1;
          }
        }
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for k in 0..len {
          for l in 0..len {
            if !has_canonical_basepair(&(k, l)) {
              continue;
            }
            let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
            if ((i, j), (k, l)) != dict_min_stack {
              continue;
            }
            regularizers2[offset] = regularizer;
            offset += 1;
          }
        }
      }
    }
    let len = self.terminal_mismatch_scores.len();
    let mut effective_group_size = 0;
    let mut squared_sum = 0.;
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for k in 0..len {
          for l in 0..len {
            let x = self.terminal_mismatch_scores[i][j][k][l];
            squared_sum += x * x;
            effective_group_size += 1;
          }
        }
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for _ in 0..len {
          for _ in 0..len {
            regularizers2[offset] = regularizer;
            offset += 1;
          }
        }
      }
    }
    let len = self.dangling_scores_left.len();
    let mut effective_group_size = 0;
    let mut squared_sum = 0.;
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for k in 0..len {
          let x = self.dangling_scores_left[i][j][k];
          squared_sum += x * x;
          effective_group_size += 1;
        }
      }
    }
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for k in 0..len {
          let x = self.dangling_scores_right[i][j][k];
          squared_sum += x * x;
          effective_group_size += 1;
        }
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for _ in 0..len {
          regularizers2[offset] = regularizer;
          offset += 1;
        }
      }
    }
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for _ in 0..len {
          regularizers2[offset] = regularizer;
          offset += 1;
        }
      }
    }
    let len = self.helix_close_scores.len();
    let mut effective_group_size = 0;
    let mut squared_sum = 0.;
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        let x = self.helix_close_scores[i][j];
        squared_sum += x * x;
        effective_group_size += 1;
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        regularizers2[offset] = regularizer;
        offset += 1;
      }
    }
    let len = self.basepair_scores.len();
    let mut effective_group_size = 0;
    let mut squared_sum = 0.;
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        let dict_min_basepair = get_dict_min_pair(&(i, j));
        if (i, j) != dict_min_basepair {
          continue;
        }
        let x = self.basepair_scores[i][j];
        squared_sum += x * x;
        effective_group_size += 1;
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        let dict_min_basepair = get_dict_min_pair(&(i, j));
        if (i, j) != dict_min_basepair {
          continue;
        }
        regularizers2[offset] = regularizer;
        offset += 1;
      }
    }
    let len = self.interior_scores_explicit.len();
    let mut effective_group_size = 0;
    let mut squared_sum = 0.;
    for i in 0..len {
      for j in 0..len {
        let dict_min_len_pair = get_dict_min_pair(&(i, j));
        if (i, j) != dict_min_len_pair {
          continue;
        }
        let x = self.interior_scores_explicit[i][j];
        squared_sum += x * x;
        effective_group_size += 1;
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0..len {
      for j in 0..len {
        let dict_min_len_pair = get_dict_min_pair(&(i, j));
        if (i, j) != dict_min_len_pair {
          continue;
        }
        regularizers2[offset] = regularizer;
        offset += 1;
      }
    }
    let len = self.bulge_scores_0x1.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0..len {
      let x = self.bulge_scores_0x1[i];
      squared_sum += x * x;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0..len {
      regularizers2[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.interior_scores_1x1.len();
    let mut effective_group_size = 0;
    let mut squared_sum = 0.;
    for i in 0..len {
      for j in 0..len {
        let dict_min_basepair = get_dict_min_pair(&(i, j));
        if (i, j) != dict_min_basepair {
          continue;
        }
        let x = self.interior_scores_1x1[i][j];
        squared_sum += x * x;
        effective_group_size += 1;
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0..len {
      for j in 0..len {
        let dict_min_basepair = get_dict_min_pair(&(i, j));
        if (i, j) != dict_min_basepair {
          continue;
        }
        regularizers2[offset] = regularizer;
        offset += 1;
      }
    }
    let mut squared_sum = 0.;
    squared_sum += self.multibranch_score_base * self.multibranch_score_base;
    squared_sum += self.multibranch_score_basepair * self.multibranch_score_basepair;
    squared_sum += self.multibranch_score_unpair * self.multibranch_score_unpair;
    let regularizer = get_regularizer(GROUP_SIZE_MULTIBRANCH, squared_sum);
    regularizers2[offset] = regularizer;
    offset += 1;
    regularizers2[offset] = regularizer;
    offset += 1;
    regularizers2[offset] = regularizer;
    offset += 1;
    let mut squared_sum = 0.;
    squared_sum += self.external_score_basepair * self.external_score_basepair;
    squared_sum += self.external_score_unpair * self.external_score_unpair;
    let regularizer = get_regularizer(GROUP_SIZE_EXTERNAL, squared_sum);
    regularizers2[offset] = regularizer;
    offset += 1;
    regularizers2[offset] = regularizer;
    offset += 1;
    let mut squared_sum = 0.;
    squared_sum += self.match2match_score * self.match2match_score;
    squared_sum += self.match2insert_score * self.match2insert_score;
    squared_sum += self.init_match_score * self.init_match_score;
    let regularizer = get_regularizer(GROUP_SIZE_MATCH_TRANSITION, squared_sum);
    regularizers2[offset] = regularizer;
    offset += 1;
    regularizers2[offset] = regularizer;
    offset += 1;
    regularizers2[offset] = regularizer;
    offset += 1;
    let mut squared_sum = 0.;
    squared_sum += self.insert_extend_score * self.insert_extend_score;
    squared_sum += self.init_insert_score * self.init_insert_score;
    let regularizer = get_regularizer(GROUP_SIZE_INSERT_TRANSITION, squared_sum);
    regularizers2[offset] = regularizer;
    offset += 1;
    regularizers2[offset] = regularizer;
    offset += 1;
    let len = self.insert_scores.len();
    let group_size = len;
    let mut squared_sum = 0.;
    for i in 0..len {
      let x = self.insert_scores[i];
      squared_sum += x * x;
    }
    let regularizer = get_regularizer(group_size, squared_sum);
    for i in 0..len {
      regularizers2[offset + i] = regularizer;
    }
    offset += group_size;
    let len = self.match_scores.len();
    let mut effective_group_size = 0;
    let mut squared_sum = 0.;
    for i in 0..len {
      for j in 0..len {
        let dict_min_match = get_dict_min_pair(&(i, j));
        if (i, j) != dict_min_match {
          continue;
        }
        let x = self.match_scores[i][j];
        squared_sum += x * x;
        effective_group_size += 1;
      }
    }
    let regularizer = get_regularizer(effective_group_size, squared_sum);
    for i in 0..len {
      for j in 0..len {
        let dict_min_match = get_dict_min_pair(&(i, j));
        if (i, j) != dict_min_match {
          continue;
        }
        regularizers2[offset] = regularizer;
        offset += 1;
      }
    }
    assert!(offset == self.len());
    *regularizers = Array1::from(regularizers2);
  }

  pub fn update<T: HashIndex>(
    &mut self,
    train_data: &[TrainDatum<T>],
    regularizers: &mut Regularizers,
  ) {
    let f = |_: &BfgsScores| self.get_cost(train_data, regularizers) as BfgsScore;
    let g = |_: &BfgsScores| scores2bfgs_scores(&self.get_grad(train_data, regularizers));
    let uses_cumulative_scores = false;
    match bfgs(
      scores2bfgs_scores(&struct2vec(self, uses_cumulative_scores)),
      f,
      g,
    ) {
      Ok(solution) => {
        *self = vec2struct(&bfgs_scores2scores(&solution), uses_cumulative_scores);
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
    for i in 0..NUM_BASES {
      for j in 0..NUM_BASES {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for k in 0..NUM_BASES {
          for l in 0..NUM_BASES {
            if !has_canonical_basepair(&(k, l)) {
              continue;
            }
            let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
            if ((i, j), (k, l)) == dict_min_stack {
              continue;
            }
            self.stack_scores[i][j][k][l] = self.stack_scores[dict_min_stack.0 .0]
              [dict_min_stack.0 .1][dict_min_stack.1 .0][dict_min_stack.1 .1];
          }
        }
      }
    }
    for i in 0..NUM_BASES {
      for j in 0..NUM_BASES {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        let dict_min_basepair = get_dict_min_pair(&(i, j));
        if (i, j) == dict_min_basepair {
          continue;
        }
        self.basepair_scores[i][j] = self.basepair_scores[dict_min_basepair.0][dict_min_basepair.1];
      }
    }
    let len = self.interior_scores_explicit.len();
    for i in 0..len {
      for j in 0..len {
        let dict_min_len_pair = get_dict_min_pair(&(i, j));
        if (i, j) == dict_min_len_pair {
          continue;
        }
        self.interior_scores_explicit[i][j] =
          self.interior_scores_explicit[dict_min_len_pair.0][dict_min_len_pair.1];
      }
    }
    for i in 0..NUM_BASES {
      for j in 0..NUM_BASES {
        let dict_min_basepair = get_dict_min_pair(&(i, j));
        if (i, j) == dict_min_basepair {
          continue;
        }
        self.interior_scores_1x1[i][j] =
          self.interior_scores_1x1[dict_min_basepair.0][dict_min_basepair.1];
      }
    }
    for i in 0..NUM_BASES {
      for j in 0..NUM_BASES {
        let dict_min_match = get_dict_min_pair(&(i, j));
        if (i, j) == dict_min_match {
          continue;
        }
        self.match_scores[i][j] = self.match_scores[dict_min_match.0][dict_min_match.1];
      }
    }
  }

  pub fn accumulate(&mut self) {
    let mut sum = 0.;
    for i in 0..self.hairpin_scores_len_cumulative.len() {
      sum += self.hairpin_scores_len[i];
      self.hairpin_scores_len_cumulative[i] = sum;
    }
    let mut sum = 0.;
    for i in 0..self.bulge_scores_len_cumulative.len() {
      sum += self.bulge_scores_len[i];
      self.bulge_scores_len_cumulative[i] = sum;
    }
    let mut sum = 0.;
    for i in 0..self.interior_scores_len_cumulative.len() {
      sum += self.interior_scores_len[i];
      self.interior_scores_len_cumulative[i] = sum;
    }
    let mut sum = 0.;
    for i in 0..self.interior_scores_symmetric_cumulative.len() {
      sum += self.interior_scores_symmetric[i];
      self.interior_scores_symmetric_cumulative[i] = sum;
    }
    let mut sum = 0.;
    for i in 0..self.interior_scores_asymmetric_cumulative.len() {
      sum += self.interior_scores_asymmetric[i];
      self.interior_scores_asymmetric_cumulative[i] = sum;
    }
  }

  pub fn get_grad<T: HashIndex>(
    &self,
    train_data: &[TrainDatum<T>],
    regularizers: &Regularizers,
  ) -> Scores {
    let uses_cumulative_scores = false;
    let alignfold_scores = struct2vec(self, uses_cumulative_scores);
    let mut grad = AlignfoldScores::new(0.);
    for train_datum in train_data {
      let obs = &train_datum.alignfold_counts_observed;
      let expect = &train_datum.alignfold_counts_expected;
      let mut sum = 0.;
      let len = obs.hairpin_scores_len.len();
      for i in (0..len).rev() {
        let x = obs.hairpin_scores_len[i];
        let y = expect.hairpin_scores_len[i];
        sum -= x - y;
        grad.hairpin_scores_len[i] += sum;
      }
      let len = obs.bulge_scores_len.len();
      let mut sum = 0.;
      for i in (0..len).rev() {
        let x = obs.bulge_scores_len[i];
        let y = expect.bulge_scores_len[i];
        sum -= x - y;
        grad.bulge_scores_len[i] += sum;
      }
      let len = obs.interior_scores_len.len();
      let mut sum = 0.;
      for i in (0..len).rev() {
        let x = obs.interior_scores_len[i];
        let y = expect.interior_scores_len[i];
        sum -= x - y;
        grad.interior_scores_len[i] += sum;
      }
      let len = obs.interior_scores_symmetric.len();
      let mut sum = 0.;
      for i in (0..len).rev() {
        let x = obs.interior_scores_symmetric[i];
        let y = expect.interior_scores_symmetric[i];
        sum -= x - y;
        grad.interior_scores_symmetric[i] += sum;
      }
      let len = obs.interior_scores_asymmetric.len();
      let mut sum = 0.;
      for i in (0..len).rev() {
        let x = obs.interior_scores_asymmetric[i];
        let y = expect.interior_scores_asymmetric[i];
        sum -= x - y;
        grad.interior_scores_asymmetric[i] += sum;
      }
      for i in 0..NUM_BASES {
        for j in 0..NUM_BASES {
          if !has_canonical_basepair(&(i, j)) {
            continue;
          }
          for k in 0..NUM_BASES {
            for l in 0..NUM_BASES {
              if !has_canonical_basepair(&(k, l)) {
                continue;
              }
              let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
              if ((i, j), (k, l)) != dict_min_stack {
                continue;
              }
              let x = obs.stack_scores[i][j][k][l];
              let y = expect.stack_scores[i][j][k][l];
              grad.stack_scores[i][j][k][l] -= x - y;
            }
          }
        }
      }
      for i in 0..NUM_BASES {
        for j in 0..NUM_BASES {
          if !has_canonical_basepair(&(i, j)) {
            continue;
          }
          for k in 0..NUM_BASES {
            for l in 0..NUM_BASES {
              let x = obs.terminal_mismatch_scores[i][j][k][l];
              let y = expect.terminal_mismatch_scores[i][j][k][l];
              grad.terminal_mismatch_scores[i][j][k][l] -= x - y;
            }
          }
        }
      }
      for i in 0..NUM_BASES {
        for j in 0..NUM_BASES {
          if !has_canonical_basepair(&(i, j)) {
            continue;
          }
          for k in 0..NUM_BASES {
            let x = obs.dangling_scores_left[i][j][k];
            let y = expect.dangling_scores_left[i][j][k];
            grad.dangling_scores_left[i][j][k] -= x - y;
          }
        }
      }
      for i in 0..NUM_BASES {
        for j in 0..NUM_BASES {
          if !has_canonical_basepair(&(i, j)) {
            continue;
          }
          for k in 0..NUM_BASES {
            let x = obs.dangling_scores_right[i][j][k];
            let y = expect.dangling_scores_right[i][j][k];
            grad.dangling_scores_right[i][j][k] -= x - y;
          }
        }
      }
      for i in 0..NUM_BASES {
        for j in 0..NUM_BASES {
          if !has_canonical_basepair(&(i, j)) {
            continue;
          }
          let x = obs.helix_close_scores[i][j];
          let y = expect.helix_close_scores[i][j];
          grad.helix_close_scores[i][j] -= x - y;
        }
      }
      for i in 0..NUM_BASES {
        for j in 0..NUM_BASES {
          if !has_canonical_basepair(&(i, j)) {
            continue;
          }
          let dict_min_basepair = get_dict_min_pair(&(i, j));
          if (i, j) != dict_min_basepair {
            continue;
          }
          let x = obs.basepair_scores[i][j];
          let y = expect.basepair_scores[i][j];
          grad.basepair_scores[i][j] -= x - y;
        }
      }
      let len = obs.interior_scores_explicit.len();
      for i in 0..len {
        for j in 0..len {
          let dict_min_len_pair = get_dict_min_pair(&(i, j));
          if (i, j) != dict_min_len_pair {
            continue;
          }
          let x = obs.interior_scores_explicit[i][j];
          let y = expect.interior_scores_explicit[i][j];
          grad.interior_scores_explicit[i][j] -= x - y;
        }
      }
      for i in 0..NUM_BASES {
        let x = obs.bulge_scores_0x1[i];
        let y = expect.bulge_scores_0x1[i];
        grad.bulge_scores_0x1[i] -= x - y;
      }
      for i in 0..NUM_BASES {
        for j in 0..NUM_BASES {
          let dict_min_basepair = get_dict_min_pair(&(i, j));
          if (i, j) != dict_min_basepair {
            continue;
          }
          let x = obs.interior_scores_1x1[i][j];
          let y = expect.interior_scores_1x1[i][j];
          grad.interior_scores_1x1[i][j] -= x - y;
        }
      }
      let obs_score = obs.multibranch_score_base;
      let expect_score = expect.multibranch_score_base;
      grad.multibranch_score_base -= obs_score - expect_score;
      let obs_score = obs.multibranch_score_basepair;
      let expect_score = expect.multibranch_score_basepair;
      grad.multibranch_score_basepair -= obs_score - expect_score;
      let obs_score = obs.multibranch_score_unpair;
      let expect_score = expect.multibranch_score_unpair;
      grad.multibranch_score_unpair -= obs_score - expect_score;
      let obs_score = obs.external_score_basepair;
      let expect_score = expect.external_score_basepair;
      grad.external_score_basepair -= obs_score - expect_score;
      let obs_score = obs.external_score_unpair;
      let expect_score = expect.external_score_unpair;
      grad.external_score_unpair -= obs_score - expect_score;
      let obs_score = obs.match2match_score;
      let expect_score = expect.match2match_score;
      grad.match2match_score -= obs_score - expect_score;
      let obs_score = obs.match2insert_score;
      let expect_score = expect.match2insert_score;
      grad.match2insert_score -= obs_score - expect_score;
      let obs_score = obs.insert_extend_score;
      let expect_score = expect.insert_extend_score;
      grad.insert_extend_score -= obs_score - expect_score;
      let obs_score = obs.init_match_score;
      let expect_score = expect.init_match_score;
      grad.init_match_score -= obs_score - expect_score;
      let obs_score = obs.init_insert_score;
      let expect_score = expect.init_insert_score;
      grad.init_insert_score -= obs_score - expect_score;
      for i in 0..NUM_BASES {
        let x = obs.insert_scores[i];
        let y = expect.insert_scores[i];
        grad.insert_scores[i] -= x - y;
      }
      for i in 0..NUM_BASES {
        for j in 0..NUM_BASES {
          let dict_min_match = get_dict_min_pair(&(i, j));
          if (i, j) != dict_min_match {
            continue;
          }
          let x = obs.match_scores[i][j];
          let y = expect.match_scores[i][j];
          grad.match_scores[i][j] -= x - y;
        }
      }
    }
    struct2vec(&grad, uses_cumulative_scores) + regularizers * &alignfold_scores
  }

  pub fn get_cost<T: HashIndex>(
    &self,
    train_data: &[TrainDatum<T>],
    regularizers: &Regularizers,
  ) -> Score {
    let uses_cumulative_scores = true;
    let mut log_likelihood = 0.;
    let alignfold_scores_cumulative = struct2vec(self, uses_cumulative_scores);
    let uses_cumulative_scores = false;
    for train_datum in train_data {
      let obs = &train_datum.alignfold_counts_observed;
      log_likelihood += alignfold_scores_cumulative.dot(&struct2vec(obs, uses_cumulative_scores));
      log_likelihood -= train_datum.global_sum;
    }
    let alignfold_scores = struct2vec(self, uses_cumulative_scores);
    let product = regularizers * &alignfold_scores;
    -log_likelihood + product.dot(&alignfold_scores) / 2.
  }

  pub fn rand_init(&mut self) {
    let len = self.len();
    let std_deviation = 1. / (len as Score).sqrt();
    let normal = Normal::new(0., std_deviation).unwrap();
    let mut thread_rng = thread_rng();
    for x in self.hairpin_scores_len.iter_mut() {
      *x = normal.sample(&mut thread_rng);
    }
    for x in self.bulge_scores_len.iter_mut() {
      *x = normal.sample(&mut thread_rng);
    }
    for x in self.interior_scores_len.iter_mut() {
      *x = normal.sample(&mut thread_rng);
    }
    for x in self.interior_scores_symmetric.iter_mut() {
      *x = normal.sample(&mut thread_rng);
    }
    for x in self.interior_scores_asymmetric.iter_mut() {
      *x = normal.sample(&mut thread_rng);
    }
    let len = self.stack_scores.len();
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for k in 0..len {
          for l in 0..len {
            if !has_canonical_basepair(&(k, l)) {
              continue;
            }
            let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
            if ((i, j), (k, l)) != dict_min_stack {
              continue;
            }
            let x = normal.sample(&mut thread_rng);
            self.stack_scores[i][j][k][l] = x;
          }
        }
      }
    }
    let len = self.terminal_mismatch_scores.len();
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for k in 0..len {
          for l in 0..len {
            let x = normal.sample(&mut thread_rng);
            self.terminal_mismatch_scores[i][j][k][l] = x;
          }
        }
      }
    }
    let len = self.dangling_scores_left.len();
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for k in 0..len {
          let x = normal.sample(&mut thread_rng);
          self.dangling_scores_left[i][j][k] = x;
        }
      }
    }
    let len = self.dangling_scores_right.len();
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for k in 0..len {
          let x = normal.sample(&mut thread_rng);
          self.dangling_scores_right[i][j][k] = x;
        }
      }
    }
    let len = self.helix_close_scores.len();
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        let x = normal.sample(&mut thread_rng);
        self.helix_close_scores[i][j] = x;
      }
    }
    let len = self.basepair_scores.len();
    for i in 0..len {
      for j in 0..len {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        let dict_min_basepair = get_dict_min_pair(&(i, j));
        if (i, j) != dict_min_basepair {
          continue;
        }
        let x = normal.sample(&mut thread_rng);
        self.basepair_scores[i][j] = x;
      }
    }
    let len = self.interior_scores_explicit.len();
    for i in 0..len {
      for j in 0..len {
        let dict_min_len_pair = get_dict_min_pair(&(i, j));
        if (i, j) != dict_min_len_pair {
          continue;
        }
        let x = normal.sample(&mut thread_rng);
        self.interior_scores_explicit[i][j] = x;
      }
    }
    for x in &mut self.bulge_scores_0x1 {
      *x = normal.sample(&mut thread_rng);
    }
    let len = self.interior_scores_1x1.len();
    for i in 0..len {
      for j in 0..len {
        let dict_min_basepair = get_dict_min_pair(&(i, j));
        if (i, j) != dict_min_basepair {
          continue;
        }
        let x = normal.sample(&mut thread_rng);
        self.interior_scores_1x1[i][j] = x;
      }
    }
    self.multibranch_score_base = normal.sample(&mut thread_rng);
    self.multibranch_score_basepair = normal.sample(&mut thread_rng);
    self.multibranch_score_unpair = normal.sample(&mut thread_rng);
    self.external_score_basepair = normal.sample(&mut thread_rng);
    self.external_score_unpair = normal.sample(&mut thread_rng);
    self.match2match_score = normal.sample(&mut thread_rng);
    self.match2insert_score = normal.sample(&mut thread_rng);
    self.insert_extend_score = normal.sample(&mut thread_rng);
    self.init_match_score = normal.sample(&mut thread_rng);
    self.init_insert_score = normal.sample(&mut thread_rng);
    let len = self.insert_scores.len();
    for i in 0..len {
      let x = normal.sample(&mut thread_rng);
      self.insert_scores[i] = x;
    }
    let len = self.match_scores.len();
    for i in 0..len {
      for j in 0..len {
        let dict_min_match = get_dict_min_pair(&(i, j));
        if (i, j) != dict_min_match {
          continue;
        }
        let x = normal.sample(&mut thread_rng);
        self.match_scores[i][j] = x;
      }
    }
    self.mirror();
    self.accumulate();
  }

  pub fn transfer(&mut self) {
    for (x, &y) in self
      .hairpin_scores_len
      .iter_mut()
      .zip(HAIRPIN_SCORES_LEN_ATLEAST.iter())
    {
      *x = y;
    }
    for (x, &y) in self
      .bulge_scores_len
      .iter_mut()
      .zip(BULGE_SCORES_LEN_ATLEAST.iter())
    {
      *x = y;
    }
    for (x, &y) in self
      .interior_scores_len
      .iter_mut()
      .zip(INTERIOR_SCORES_LEN_ATLEAST.iter())
    {
      *x = y;
    }
    for (x, &y) in self
      .interior_scores_symmetric
      .iter_mut()
      .zip(INTERIOR_SCORES_SYMMETRIC_ATLEAST.iter())
    {
      *x = y;
    }
    for (x, &y) in self
      .interior_scores_asymmetric
      .iter_mut()
      .zip(INTERIOR_SCORES_ASYMMETRIC_ATLEAST.iter())
    {
      *x = y;
    }
    for (i, x) in STACK_SCORES_CONTRA.iter().enumerate() {
      for (j, x) in x.iter().enumerate() {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for (k, x) in x.iter().enumerate() {
          for (l, &x) in x.iter().enumerate() {
            if !has_canonical_basepair(&(k, l)) {
              continue;
            }
            self.stack_scores[i][j][k][l] = x;
          }
        }
      }
    }
    for (i, x) in TERMINAL_MISMATCH_SCORES_CONTRA.iter().enumerate() {
      for (j, x) in x.iter().enumerate() {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for (k, x) in x.iter().enumerate() {
          for (l, &x) in x.iter().enumerate() {
            self.terminal_mismatch_scores[i][j][k][l] = x;
          }
        }
      }
    }
    for (i, x) in DANGLING_SCORES_LEFT.iter().enumerate() {
      for (j, x) in x.iter().enumerate() {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for (k, &x) in x.iter().enumerate() {
          self.dangling_scores_left[i][j][k] = x;
        }
      }
    }
    for (i, x) in DANGLING_SCORES_RIGHT.iter().enumerate() {
      for (j, x) in x.iter().enumerate() {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        for (k, &x) in x.iter().enumerate() {
          self.dangling_scores_right[i][j][k] = x;
        }
      }
    }
    for (i, x) in HELIX_CLOSE_SCORES.iter().enumerate() {
      for (j, &x) in x.iter().enumerate() {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        self.helix_close_scores[i][j] = x;
      }
    }
    for (i, x) in BASEPAIR_SCORES.iter().enumerate() {
      for (j, &x) in x.iter().enumerate() {
        if !has_canonical_basepair(&(i, j)) {
          continue;
        }
        self.basepair_scores[i][j] = x;
      }
    }
    for (i, x) in INTERIOR_SCORES_EXPLICIT.iter().enumerate() {
      for (j, &x) in x.iter().enumerate() {
        self.interior_scores_explicit[i][j] = x;
      }
    }
    for (x, &y) in self
      .bulge_scores_0x1
      .iter_mut()
      .zip(BULGE_SCORES_0X1.iter())
    {
      *x = y;
    }
    for (i, x) in INTERIOR_SCORES_1X1_CONTRA.iter().enumerate() {
      for (j, &x) in x.iter().enumerate() {
        self.interior_scores_1x1[i][j] = x;
      }
    }
    self.multibranch_score_base = MULTIBRANCH_SCORE_BASE;
    self.multibranch_score_basepair = MULTIBRANCH_SCORE_BASEPAIR;
    self.multibranch_score_unpair = MULTIBRANCH_SCORE_UNPAIR;
    self.external_score_basepair = EXTERNAL_SCORE_BASEPAIR;
    self.external_score_unpair = EXTERNAL_SCORE_UNPAIR;
    self.match2match_score = MATCH2MATCH_SCORE;
    self.match2insert_score = MATCH2INSERT_SCORE;
    self.insert_extend_score = INSERT_EXTEND_SCORE;
    self.init_match_score = INIT_MATCH_SCORE;
    self.init_insert_score = INIT_INSERT_SCORE;
    for (i, &x) in INSERT_SCORES.iter().enumerate() {
      self.insert_scores[i] = x;
    }
    for (i, x) in MATCH_SCORES.iter().enumerate() {
      for (j, &x) in x.iter().enumerate() {
        self.match_scores[i][j] = x;
      }
    }
    self.accumulate();
  }
}

impl<T: HashIndex> TrainDatum<T> {
  pub fn origin() -> TrainDatum<T> {
    TrainDatum {
      seq_pair: (Seq::new(), Seq::new()),
      seq_pair_gapped: (Seq::new(), Seq::new()),
      alignfold_counts_observed: AlignfoldScores::new(0.),
      alignfold_counts_expected: AlignfoldScores::new(NEG_INFINITY),
      basepair_probs_pair: (SparseProbMat::<T>::default(), SparseProbMat::<T>::default()),
      max_basepair_span_pair: (T::zero(), T::zero()),
      global_sum: NEG_INFINITY,
      forward_pos_pairs: PosPairMatSet::<T>::default(),
      backward_pos_pairs: PosPairMatSet::<T>::default(),
      pos_quads_hashed_lens: PosQuadsHashedLens::<T>::default(),
      matchable_poss: SparsePosSets::<T>::default(),
      matchable_poss2: SparsePosSets::<T>::default(),
      fold_scores_pair: (FoldScoresTrained::<T>::new(), FoldScoresTrained::<T>::new()),
      match_probs: SparseProbMat::<T>::default(),
      alignfold: PairAlignfold::<T>::new(),
      accuracy: NEG_INFINITY,
    }
  }

  pub fn new(
    input_file_path: &Path,
    min_basepair_prob: Prob,
    min_match_prob: Prob,
    align_scores: &AlignScores,
  ) -> TrainDatum<T> {
    let fasta_file_reader = Reader::from_file(Path::new(input_file_path)).unwrap();
    let fasta_records: Vec<Record> = fasta_file_reader
      .records()
      .map(|rec| rec.unwrap())
      .collect();
    let consensus_fold = fasta_records[2].seq();
    let seq_pair = (
      bytes2seq_gapped(fasta_records[0].seq()),
      bytes2seq_gapped(fasta_records[1].seq()),
    );
    let mut seq_pair_ungapped = (gapped2ungapped(&seq_pair.0), gapped2ungapped(&seq_pair.1));
    let uses_contra_model = false;
    let allows_short_hairpins = true;
    let basepair_probs_pair = (
      filter_basepair_probs::<T>(
        &mccaskill_algo(
          &seq_pair_ungapped.0[..],
          uses_contra_model,
          allows_short_hairpins,
          &FoldScoreSets::new(0.),
        )
        .0,
        min_basepair_prob,
      ),
      filter_basepair_probs::<T>(
        &mccaskill_algo(
          &seq_pair_ungapped.1[..],
          uses_contra_model,
          allows_short_hairpins,
          &FoldScoreSets::new(0.),
        )
        .0,
        min_basepair_prob,
      ),
    );
    seq_pair_ungapped.0.insert(0, PSEUDO_BASE);
    seq_pair_ungapped.0.push(PSEUDO_BASE);
    seq_pair_ungapped.1.insert(0, PSEUDO_BASE);
    seq_pair_ungapped.1.push(PSEUDO_BASE);
    let seq_len_pair = (
      T::from_usize(seq_pair_ungapped.0.len()).unwrap(),
      T::from_usize(seq_pair_ungapped.1.len()).unwrap(),
    );
    let match_probs = filter_match_probs(
      &durbin_algo(
        &(&seq_pair_ungapped.0[..], &seq_pair_ungapped.1[..]),
        align_scores,
      ),
      min_match_prob,
    );
    let (
      forward_pos_pairs,
      backward_pos_pairs,
      _,
      pos_quads_hashed_lens,
      matchable_poss,
      matchable_poss2,
    ) = get_sparse_poss(
      &(&basepair_probs_pair.0, &basepair_probs_pair.1),
      &match_probs,
      &seq_len_pair,
    );
    let max_basepair_span_pair = (
      get_max_basepair_span::<T>(&basepair_probs_pair.0),
      get_max_basepair_span::<T>(&basepair_probs_pair.1),
    );
    let mut train_datum = TrainDatum {
      seq_pair: seq_pair_ungapped,
      seq_pair_gapped: seq_pair,
      alignfold_counts_observed: AlignfoldScores::new(0.),
      alignfold_counts_expected: AlignfoldScores::new(NEG_INFINITY),
      basepair_probs_pair,
      max_basepair_span_pair,
      global_sum: NEG_INFINITY,
      forward_pos_pairs,
      backward_pos_pairs,
      pos_quads_hashed_lens,
      matchable_poss,
      matchable_poss2,
      fold_scores_pair: (FoldScoresTrained::<T>::new(), FoldScoresTrained::<T>::new()),
      match_probs,
      alignfold: PairAlignfold::<T>::new(),
      accuracy: NEG_INFINITY,
    };
    train_datum.obs2counts(consensus_fold);
    train_datum
  }

  pub fn obs2counts(&mut self, consensus_fold: TextSlice) {
    let align_len = consensus_fold.len();
    let mut inserted = false;
    let mut inserted2 = inserted;
    let seq_pair = &self.seq_pair_gapped;
    let mut pos_pair = (T::one(), T::one());
    for (i, &notation) in consensus_fold.iter().enumerate() {
      let basepair = (seq_pair.0[i], seq_pair.1[i]);
      if notation != UNPAIR {
        let dict_min_match = get_dict_min_pair(&basepair);
        self.alignfold_counts_observed.match_scores[dict_min_match.0][dict_min_match.1] += 1.;
        if i == 0 {
          self.alignfold_counts_observed.init_match_score += 1.;
        } else if inserted || inserted2 {
          self.alignfold_counts_observed.match2insert_score += 1.;
        } else {
          self.alignfold_counts_observed.match2match_score += 1.;
        }
        inserted = false;
        inserted2 = inserted;
        if basepair.1 == PSEUDO_BASE {
          self.alignfold.inserted_poss.insert(pos_pair.0);
          pos_pair.0 = pos_pair.0 + T::one();
        } else if basepair.0 == PSEUDO_BASE {
          self.alignfold.deleted_poss.insert(pos_pair.1);
          pos_pair.1 = pos_pair.1 + T::one();
        } else {
          self.alignfold.matched_pos_pairs.insert(pos_pair);
          pos_pair.0 = pos_pair.0 + T::one();
          pos_pair.1 = pos_pair.1 + T::one();
        }
        continue;
      }
      if basepair.1 == PSEUDO_BASE {
        if i == 0 {
          self.alignfold_counts_observed.init_insert_score += 1.;
        } else if inserted {
          self.alignfold_counts_observed.insert_extend_score += 1.;
        } else if inserted2 {
          inserted = true;
          inserted2 = false;
        } else {
          self.alignfold_counts_observed.match2insert_score += 1.;
          inserted = true;
        }
        self.alignfold_counts_observed.insert_scores[basepair.0] += 1.;
        self.alignfold.inserted_poss.insert(pos_pair.0);
        pos_pair.0 = pos_pair.0 + T::one();
      } else if basepair.0 == PSEUDO_BASE {
        if i == 0 {
          self.alignfold_counts_observed.init_insert_score += 1.;
          inserted2 = true;
        } else if inserted2 {
          self.alignfold_counts_observed.insert_extend_score += 1.;
        } else if inserted {
          inserted2 = true;
          inserted = false;
        } else {
          self.alignfold_counts_observed.match2insert_score += 1.;
          inserted2 = true;
        }
        self.alignfold_counts_observed.insert_scores[basepair.1] += 1.;
        self.alignfold.deleted_poss.insert(pos_pair.1);
        pos_pair.1 = pos_pair.1 + T::one();
      } else {
        let dict_min_match = get_dict_min_pair(&basepair);
        self.alignfold_counts_observed.match_scores[dict_min_match.0][dict_min_match.1] += 1.;
        if i == 0 {
          self.alignfold_counts_observed.init_match_score += 1.;
        } else if inserted || inserted2 {
          self.alignfold_counts_observed.match2insert_score += 1.;
        } else {
          self.alignfold_counts_observed.match2match_score += 1.;
        }
        inserted = false;
        inserted2 = inserted;
        self.alignfold.matched_pos_pairs.insert(pos_pair);
        pos_pair.0 = pos_pair.0 + T::one();
        pos_pair.1 = pos_pair.1 + T::one();
      }
    }
    let mut stack = Vec::new();
    let mut consensus_basepairs = HashSet::<(usize, usize)>::default();
    for (i, &x) in consensus_fold.iter().enumerate() {
      if x == BASEPAIR_LEFT {
        stack.push(i);
      } else if x == BASEPAIR_RIGHT {
        let x = stack.pop().unwrap();
        consensus_basepairs.insert((x, i));
        let y = (seq_pair.0[x], seq_pair.0[i]);
        if has_canonical_basepair(&y) {
          let y = get_dict_min_pair(&y);
          self.alignfold_counts_observed.basepair_scores[y.0][y.1] += 1.;
        }
        let y = (seq_pair.1[x], seq_pair.1[i]);
        if has_canonical_basepair(&y) {
          let y = get_dict_min_pair(&y);
          self.alignfold_counts_observed.basepair_scores[y.0][y.1] += 1.;
        }
      }
    }
    let mut loop_struct = LoopStruct::default();
    let mut stored_basepairs = HashSet::<(usize, usize)>::default();
    for substr_len in 2..align_len + 1 {
      for i in 0..align_len - substr_len + 1 {
        let mut found_basepair = false;
        let j = i + substr_len - 1;
        if consensus_basepairs.contains(&(i, j)) {
          found_basepair = true;
          consensus_basepairs.remove(&(i, j));
          stored_basepairs.insert((i, j));
        }
        if found_basepair {
          let mut loop_basepairs = Vec::new();
          for x in stored_basepairs.iter() {
            if i < x.0 && x.1 < j {
              loop_basepairs.push(*x);
            }
          }
          for x in loop_basepairs.iter() {
            stored_basepairs.remove(x);
          }
          loop_basepairs.sort();
          loop_struct.insert((i, j), loop_basepairs);
        }
      }
    }
    for (basepair_close, basepairs_loop) in loop_struct.iter() {
      let num_basepairs_loop = basepairs_loop.len();
      let basepair = (seq_pair.0[basepair_close.0], seq_pair.0[basepair_close.1]);
      let basepair2 = (seq_pair.1[basepair_close.0], seq_pair.1[basepair_close.1]);
      let closes = true;
      let mismatch_pair = get_mismatch_pair(&seq_pair.0[..], basepair_close, closes);
      let mismatch_pair2 = get_mismatch_pair(&seq_pair.1[..], basepair_close, closes);
      if num_basepairs_loop == 0 {
        let hairpin_len_pair = (
          get_hairpin_len(&seq_pair.0[..], basepair_close),
          get_hairpin_len(&seq_pair.1[..], basepair_close),
        );
        if has_canonical_basepair(&basepair) {
          self.alignfold_counts_observed.terminal_mismatch_scores[basepair.0][basepair.1]
            [mismatch_pair.0][mismatch_pair.1] += 1.;
          self.alignfold_counts_observed.helix_close_scores[basepair.0][basepair.1] += 1.;
        }
        if has_canonical_basepair(&basepair2) {
          self.alignfold_counts_observed.terminal_mismatch_scores[basepair2.0][basepair2.1]
            [mismatch_pair2.0][mismatch_pair2.1] += 1.;
          self.alignfold_counts_observed.helix_close_scores[basepair2.0][basepair2.1] += 1.;
        }
        if hairpin_len_pair.0 <= MAX_LOOP_LEN {
          self.alignfold_counts_observed.hairpin_scores_len[hairpin_len_pair.0] += 1.;
        } else {
          self.alignfold_counts_observed.hairpin_scores_len[MAX_LOOP_LEN] += 1.;
        }
        if hairpin_len_pair.1 <= MAX_LOOP_LEN {
          self.alignfold_counts_observed.hairpin_scores_len[hairpin_len_pair.1] += 1.;
        } else {
          self.alignfold_counts_observed.hairpin_scores_len[MAX_LOOP_LEN] += 1.;
        }
      } else if num_basepairs_loop == 1 {
        let basepair_loop = &basepairs_loop[0];
        let basepair3 = (seq_pair.0[basepair_loop.0], seq_pair.0[basepair_loop.1]);
        let basepair4 = (seq_pair.1[basepair_loop.0], seq_pair.1[basepair_loop.1]);
        let twoloop_len_pair = get_2loop_len_pair(&seq_pair.0[..], basepair_close, basepair_loop);
        let sum = twoloop_len_pair.0 + twoloop_len_pair.1;
        let twoloop_len_pair2 = get_2loop_len_pair(&seq_pair.1[..], basepair_close, basepair_loop);
        let sum2 = twoloop_len_pair2.0 + twoloop_len_pair2.1;
        let closes = false;
        let mismatch_pair3 = get_mismatch_pair(&seq_pair.0[..], basepair_loop, closes);
        let mismatch_pair4 = get_mismatch_pair(&seq_pair.1[..], basepair_loop, closes);
        if sum == 0 {
          if has_canonical_basepair(&basepair) && has_canonical_basepair(&basepair3) {
            let dict_min_stack = get_dict_min_stack(&basepair, &basepair3);
            self.alignfold_counts_observed.stack_scores[dict_min_stack.0 .0]
              [dict_min_stack.0 .1][dict_min_stack.1 .0][dict_min_stack.1 .1] += 1.;
          }
        } else {
          if twoloop_len_pair.0 == 0 || twoloop_len_pair.1 == 0 {
            if sum <= MAX_LOOP_LEN {
              self.alignfold_counts_observed.bulge_scores_len[sum - 1] += 1.;
              if sum == 1 {
                let mismatch = if twoloop_len_pair.0 == 0 {
                  mismatch_pair.1
                } else {
                  mismatch_pair.0
                };
                self.alignfold_counts_observed.bulge_scores_0x1[mismatch] += 1.;
              }
            } else {
              self.alignfold_counts_observed.bulge_scores_len[MAX_LOOP_LEN - 1] += 1.;
            }
          } else {
            let diff = get_diff(twoloop_len_pair.0, twoloop_len_pair.1);
            if sum <= MAX_LOOP_LEN {
              self.alignfold_counts_observed.interior_scores_len[sum - 2] += 1.;
              if diff == 0 {
                self.alignfold_counts_observed.interior_scores_symmetric[twoloop_len_pair.0 - 1] +=
                  1.;
              } else {
                self.alignfold_counts_observed.interior_scores_asymmetric[diff - 1] += 1.;
              }
              if twoloop_len_pair.0 == 1 && twoloop_len_pair.1 == 1 {
                let dict_min_mismatch_pair = get_dict_min_pair(&mismatch_pair);
                self.alignfold_counts_observed.interior_scores_1x1[dict_min_mismatch_pair.0]
                  [dict_min_mismatch_pair.1] += 1.;
              }
              if twoloop_len_pair.0 <= MAX_INTERIOR_EXPLICIT
                && twoloop_len_pair.1 <= MAX_INTERIOR_EXPLICIT
              {
                let dict_min_len_pair = get_dict_min_pair(&twoloop_len_pair);
                self.alignfold_counts_observed.interior_scores_explicit[dict_min_len_pair.0 - 1]
                  [dict_min_len_pair.1 - 1] += 1.;
              }
            } else {
              self.alignfold_counts_observed.interior_scores_len[MAX_LOOP_LEN - 2] += 1.;
              if diff == 0 {
                if twoloop_len_pair.0 <= MAX_INTERIOR_SYMMETRIC {
                  self.alignfold_counts_observed.interior_scores_symmetric
                    [twoloop_len_pair.0 - 1] += 1.;
                } else {
                  self.alignfold_counts_observed.interior_scores_symmetric
                    [MAX_INTERIOR_SYMMETRIC - 1] += 1.;
                }
              } else if diff <= MAX_INTERIOR_ASYMMETRIC {
                self.alignfold_counts_observed.interior_scores_asymmetric[diff - 1] += 1.;
              } else {
                self.alignfold_counts_observed.interior_scores_asymmetric
                  [MAX_INTERIOR_ASYMMETRIC - 1] += 1.;
              }
            }
          }
          if has_canonical_basepair(&basepair) {
            self.alignfold_counts_observed.terminal_mismatch_scores[basepair.0][basepair.1]
              [mismatch_pair.0][mismatch_pair.1] += 1.;
            self.alignfold_counts_observed.helix_close_scores[basepair.0][basepair.1] += 1.;
          }
          if has_canonical_basepair(&basepair3) {
            self.alignfold_counts_observed.terminal_mismatch_scores[basepair3.1][basepair3.0]
              [mismatch_pair3.1][mismatch_pair3.0] += 1.;
            self.alignfold_counts_observed.helix_close_scores[basepair3.1][basepair3.0] += 1.;
          }
        }
        if sum2 == 0 {
          if has_canonical_basepair(&basepair2) && has_canonical_basepair(&basepair4) {
            let dict_min_stack2 = get_dict_min_stack(&basepair2, &basepair4);
            self.alignfold_counts_observed.stack_scores[dict_min_stack2.0 .0]
              [dict_min_stack2.0 .1][dict_min_stack2.1 .0][dict_min_stack2.1 .1] += 1.;
          }
        } else {
          if twoloop_len_pair2.0 == 0 || twoloop_len_pair2.1 == 0 {
            if sum2 <= MAX_LOOP_LEN {
              self.alignfold_counts_observed.bulge_scores_len[sum2 - 1] += 1.;
              if sum2 == 1 {
                let mismatch2 = if twoloop_len_pair2.0 == 0 {
                  mismatch_pair2.1
                } else {
                  mismatch_pair2.0
                };
                self.alignfold_counts_observed.bulge_scores_0x1[mismatch2] += 1.;
              }
            } else {
              self.alignfold_counts_observed.bulge_scores_len[MAX_LOOP_LEN - 1] += 1.;
            }
          } else {
            let diff2 = get_diff(twoloop_len_pair2.0, twoloop_len_pair2.1);
            if sum2 <= MAX_LOOP_LEN {
              self.alignfold_counts_observed.interior_scores_len[sum2 - 2] += 1.;
              if diff2 == 0 {
                self.alignfold_counts_observed.interior_scores_symmetric
                  [twoloop_len_pair2.0 - 1] += 1.;
              } else {
                self.alignfold_counts_observed.interior_scores_asymmetric[diff2 - 1] += 1.;
              }
              if twoloop_len_pair2.0 == 1 && twoloop_len_pair2.1 == 1 {
                let dict_min_mismatch_pair2 = get_dict_min_pair(&mismatch_pair2);
                self.alignfold_counts_observed.interior_scores_1x1[dict_min_mismatch_pair2.0]
                  [dict_min_mismatch_pair2.1] += 1.;
              }
              if twoloop_len_pair2.0 <= MAX_INTERIOR_EXPLICIT
                && twoloop_len_pair2.1 <= MAX_INTERIOR_EXPLICIT
              {
                let dict_min_len_pair2 = get_dict_min_pair(&twoloop_len_pair2);
                self.alignfold_counts_observed.interior_scores_explicit
                  [dict_min_len_pair2.0 - 1][dict_min_len_pair2.1 - 1] += 1.;
              }
            } else {
              self.alignfold_counts_observed.interior_scores_len[MAX_LOOP_LEN - 2] += 1.;
              if diff2 == 0 {
                if twoloop_len_pair2.0 <= MAX_INTERIOR_SYMMETRIC {
                  self.alignfold_counts_observed.interior_scores_symmetric
                    [twoloop_len_pair2.0 - 1] += 1.;
                } else {
                  self.alignfold_counts_observed.interior_scores_symmetric
                    [MAX_INTERIOR_SYMMETRIC - 1] += 1.;
                }
              } else if diff2 <= MAX_INTERIOR_ASYMMETRIC {
                self.alignfold_counts_observed.interior_scores_asymmetric[diff2 - 1] += 1.;
              } else {
                self.alignfold_counts_observed.interior_scores_asymmetric
                  [MAX_INTERIOR_ASYMMETRIC - 1] += 1.;
              }
            }
          }
          if has_canonical_basepair(&basepair2) {
            self.alignfold_counts_observed.terminal_mismatch_scores[basepair2.0][basepair2.1]
              [mismatch_pair2.0][mismatch_pair2.1] += 1.;
            self.alignfold_counts_observed.helix_close_scores[basepair2.0][basepair2.1] += 1.;
          }
          if has_canonical_basepair(&basepair4) {
            self.alignfold_counts_observed.terminal_mismatch_scores[basepair4.1][basepair4.0]
              [mismatch_pair4.1][mismatch_pair4.0] += 1.;
            self.alignfold_counts_observed.helix_close_scores[basepair4.1][basepair4.0] += 1.;
          }
        }
      } else {
        if has_canonical_basepair(&basepair) {
          self.alignfold_counts_observed.dangling_scores_left[basepair.0][basepair.1]
            [mismatch_pair.0] += 1.;
          self.alignfold_counts_observed.dangling_scores_right[basepair.0][basepair.1]
            [mismatch_pair.1] += 1.;
          self.alignfold_counts_observed.helix_close_scores[basepair.0][basepair.1] += 1.;
        }
        if has_canonical_basepair(&basepair2) {
          self.alignfold_counts_observed.dangling_scores_left[basepair2.0][basepair2.1]
            [mismatch_pair2.0] += 1.;
          self.alignfold_counts_observed.dangling_scores_right[basepair2.0][basepair2.1]
            [mismatch_pair2.1] += 1.;
          self.alignfold_counts_observed.helix_close_scores[basepair2.0][basepair2.1] += 1.;
        }
        self.alignfold_counts_observed.multibranch_score_base += 2.;
        self.alignfold_counts_observed.multibranch_score_basepair += 2.;
        self.alignfold_counts_observed.multibranch_score_basepair +=
          2. * num_basepairs_loop as Prob;
        let num_unpairs_multibranch =
          get_num_unpairs_multibranch(basepair_close, basepairs_loop, &seq_pair.0[..]);
        self.alignfold_counts_observed.multibranch_score_unpair += num_unpairs_multibranch as Prob;
        let num_unpairs_multibranch2 =
          get_num_unpairs_multibranch(basepair_close, basepairs_loop, &seq_pair.1[..]);
        self.alignfold_counts_observed.multibranch_score_unpair += num_unpairs_multibranch2 as Prob;
        for basepair_loop in basepairs_loop.iter() {
          let basepair3 = (seq_pair.0[basepair_loop.0], seq_pair.0[basepair_loop.1]);
          let closes = false;
          let mismatch_pair3 = get_mismatch_pair(&seq_pair.0[..], basepair_loop, closes);
          let basepair4 = (seq_pair.1[basepair_loop.0], seq_pair.1[basepair_loop.1]);
          let mismatch_pair4 = get_mismatch_pair(&seq_pair.1[..], basepair_loop, closes);
          if has_canonical_basepair(&basepair3) {
            self.alignfold_counts_observed.dangling_scores_left[basepair3.1][basepair3.0]
              [mismatch_pair3.1] += 1.;
            self.alignfold_counts_observed.dangling_scores_right[basepair3.1][basepair3.0]
              [mismatch_pair3.0] += 1.;
            self.alignfold_counts_observed.helix_close_scores[basepair3.1][basepair3.0] += 1.;
          }
          if has_canonical_basepair(&basepair4) {
            self.alignfold_counts_observed.dangling_scores_left[basepair4.1][basepair4.0]
              [mismatch_pair4.1] += 1.;
            self.alignfold_counts_observed.dangling_scores_right[basepair4.1][basepair4.0]
              [mismatch_pair4.0] += 1.;
            self.alignfold_counts_observed.helix_close_scores[basepair4.1][basepair4.0] += 1.;
          }
        }
      }
    }
    self.alignfold_counts_observed.external_score_basepair += 2. * stored_basepairs.len() as Prob;
    let mut stored_basepairs_sorted = stored_basepairs
      .iter()
      .copied()
      .collect::<Vec<(usize, usize)>>();
    stored_basepairs_sorted.sort();
    let num_unpairs_external = get_num_unpairs_external(&stored_basepairs_sorted, &seq_pair.0[..]);
    self.alignfold_counts_observed.external_score_unpair += num_unpairs_external as Prob;
    let num_unpairs_external2 = get_num_unpairs_external(&stored_basepairs_sorted, &seq_pair.1[..]);
    self.alignfold_counts_observed.external_score_unpair += num_unpairs_external2 as Prob;
    for basepair_loop in stored_basepairs.iter() {
      let basepair = (seq_pair.0[basepair_loop.0], seq_pair.0[basepair_loop.1]);
      let basepair2 = (seq_pair.1[basepair_loop.0], seq_pair.1[basepair_loop.1]);
      let closes = false;
      let mismatch_pair = get_mismatch_pair(&seq_pair.0[..], basepair_loop, closes);
      let mismatch_pair2 = get_mismatch_pair(&seq_pair.1[..], basepair_loop, closes);
      if has_canonical_basepair(&basepair) {
        if mismatch_pair.1 != PSEUDO_BASE {
          self.alignfold_counts_observed.dangling_scores_left[basepair.1][basepair.0]
            [mismatch_pair.1] += 1.;
        }
        if mismatch_pair.0 != PSEUDO_BASE {
          self.alignfold_counts_observed.dangling_scores_right[basepair.1][basepair.0]
            [mismatch_pair.0] += 1.;
        }
        self.alignfold_counts_observed.helix_close_scores[basepair.1][basepair.0] += 1.;
      }
      if has_canonical_basepair(&basepair2) {
        if mismatch_pair2.1 != PSEUDO_BASE {
          self.alignfold_counts_observed.dangling_scores_left[basepair2.1][basepair2.0]
            [mismatch_pair2.1] += 1.;
        }
        if mismatch_pair2.0 != PSEUDO_BASE {
          self.alignfold_counts_observed.dangling_scores_right[basepair2.1][basepair2.0]
            [mismatch_pair2.0] += 1.;
        }
        self.alignfold_counts_observed.helix_close_scores[basepair2.1][basepair2.0] += 1.;
      }
    }
  }

  pub fn set_curr_scores(&mut self, alignfold_scores: &AlignfoldScores) {
    self.fold_scores_pair.0 = FoldScoresTrained::<T>::set_curr_scores(
      alignfold_scores,
      &self.seq_pair.0,
      &self.basepair_probs_pair.0,
    );
    self.fold_scores_pair.1 = FoldScoresTrained::<T>::set_curr_scores(
      alignfold_scores,
      &self.seq_pair.1,
      &self.basepair_probs_pair.1,
    );
  }
}

impl RangeInsertScores {
  pub fn origin() -> RangeInsertScores {
    let x = Vec::new();
    RangeInsertScores {
      insert_scores: x.clone(),
      insert_scores_external: x.clone(),
      insert_scores_multibranch: x.clone(),
      insert_scores2: x.clone(),
      insert_scores_external2: x.clone(),
      insert_scores_multibranch2: x,
    }
  }

  pub fn new(seq_pair: &SeqPair, alignfold_scores: &AlignfoldScores) -> RangeInsertScores {
    let seq_len_pair = (seq_pair.0.len(), seq_pair.1.len());
    let mut range_insert_scores = RangeInsertScores::origin();
    let neg_infs = vec![
      vec![NEG_INFINITY; seq_len_pair.0.to_usize().unwrap()];
      seq_len_pair.0.to_usize().unwrap()
    ];
    range_insert_scores.insert_scores = neg_infs.clone();
    range_insert_scores.insert_scores_external = neg_infs.clone();
    range_insert_scores.insert_scores_multibranch = neg_infs;
    let neg_infs = vec![
      vec![NEG_INFINITY; seq_len_pair.1.to_usize().unwrap()];
      seq_len_pair.1.to_usize().unwrap()
    ];
    range_insert_scores.insert_scores2 = neg_infs.clone();
    range_insert_scores.insert_scores_external2 = neg_infs.clone();
    range_insert_scores.insert_scores_multibranch2 = neg_infs;
    for i in 1..seq_len_pair.1 - 1 {
      let base = seq_pair.1[i];
      let mut sum = alignfold_scores.insert_scores[base];
      let mut sum_external = sum + alignfold_scores.external_score_unpair;
      let mut sum_multibranch = sum + alignfold_scores.multibranch_score_unpair;
      range_insert_scores.insert_scores2[i][i] = sum;
      range_insert_scores.insert_scores_external2[i][i] = sum_external;
      range_insert_scores.insert_scores_multibranch2[i][i] = sum_multibranch;
      for j in i + 1..seq_len_pair.1 - 1 {
        let x = seq_pair.1[j];
        let x = alignfold_scores.insert_scores[x] + alignfold_scores.insert_extend_score;
        sum += x;
        sum_external += x + alignfold_scores.external_score_unpair;
        sum_multibranch += x + alignfold_scores.multibranch_score_unpair;
        range_insert_scores.insert_scores2[i][j] = sum;
        range_insert_scores.insert_scores_external2[i][j] = sum_external;
        range_insert_scores.insert_scores_multibranch2[i][j] = sum_multibranch;
      }
    }
    for i in 1..seq_len_pair.0 - 1 {
      let base = seq_pair.0[i];
      let term = alignfold_scores.insert_scores[base];
      let mut sum = term;
      let mut sum_external = sum + alignfold_scores.external_score_unpair;
      let mut sum_multibranch = sum + alignfold_scores.multibranch_score_unpair;
      range_insert_scores.insert_scores[i][i] = sum;
      range_insert_scores.insert_scores_external[i][i] = sum_external;
      range_insert_scores.insert_scores_multibranch[i][i] = sum_multibranch;
      for j in i + 1..seq_len_pair.0 - 1 {
        let x = seq_pair.0[j];
        let x = alignfold_scores.insert_scores[x] + alignfold_scores.insert_extend_score;
        sum += x;
        sum_external += x + alignfold_scores.external_score_unpair;
        sum_multibranch += x + alignfold_scores.multibranch_score_unpair;
        range_insert_scores.insert_scores[i][j] = sum;
        range_insert_scores.insert_scores_external[i][j] = sum_external;
        range_insert_scores.insert_scores_multibranch[i][j] = sum_multibranch;
      }
    }
    range_insert_scores
  }
}

impl<T: HashIndex> Default for PairAlignfold<T> {
  fn default() -> Self {
    Self::new()
  }
}

impl<T: HashIndex> PairAlignfold<T> {
  pub fn new() -> PairAlignfold<T> {
    PairAlignfold {
      matched_pos_pairs: SparsePosMat::<T>::default(),
      inserted_poss: SparsePoss::<T>::default(),
      deleted_poss: SparsePoss::<T>::default(),
    }
  }
}

pub const DEFAULT_BASEPAIR_PROB_TRAIN: Prob = DEFAULT_MIN_BASEPAIR_PROB;
pub const DEFAULT_MATCH_PROB_TRAIN: Prob = DEFAULT_MIN_MATCH_PROB;
pub const NUM_BASEPAIRS: usize = 6;
pub const GROUP_SIZE_MULTIBRANCH: usize = 3;
pub const GROUP_SIZE_EXTERNAL: usize = 2;
pub const GROUP_SIZE_MATCH_TRANSITION: usize = 3;
pub const GROUP_SIZE_INSERT_TRANSITION: usize = 2;
pub const GAMMA_DISTRO_ALPHA: Score = 0.;
pub const GAMMA_DISTRO_BETA: Score = 1.;
pub const DEFAULT_LEARNING_TOLERANCE: Score = 0.000_1;
pub const TRAINED_SCORES_FILE: &str = "../src/trained_alignfold_scores.rs";
pub const TRAINED_SCORES_FILE_RANDINIT: &str = "../src/trained_alignfold_scores_randinit.rs";
#[derive(Clone, Copy)]
pub enum TrainType {
  TrainedTransfer,
  TrainedRandinit,
  TransferredOnly,
}
pub const DEFAULT_TRAIN_TYPE: &str = "trained_transfer";

pub fn get_accuracy_expected<T>(
  seq_pair: &SeqPair,
  alignfold: &PairAlignfold<T>,
  match_probs: &SparseProbMat<T>,
) -> Score
where
  T: HashIndex,
{
  let seq_len_pair = (seq_pair.0.len(), seq_pair.1.len());
  let mut insert_probs_pair = (vec![1.; seq_len_pair.0], vec![1.; seq_len_pair.1]);
  for (x, &y) in match_probs {
    let x = (x.0.to_usize().unwrap(), x.1.to_usize().unwrap());
    insert_probs_pair.0[x.0] -= y;
    insert_probs_pair.1[x.1] -= y;
  }
  let total = alignfold.matched_pos_pairs.len()
    + alignfold.inserted_poss.len()
    + alignfold.deleted_poss.len();
  let mut total_expected = alignfold
    .matched_pos_pairs
    .iter()
    .map(|x| match match_probs.get(x) {
      Some(&x) => x,
      None => 0.,
    })
    .sum::<Score>();
  total_expected += alignfold
    .inserted_poss
    .iter()
    .map(|&x| insert_probs_pair.0[x.to_usize().unwrap()])
    .sum::<Score>();
  total_expected += alignfold
    .deleted_poss
    .iter()
    .map(|&x| insert_probs_pair.1[x.to_usize().unwrap()])
    .sum::<Score>();
  total_expected / total as Score
}

pub fn consprob_core<T>(inputs: InputsConsprobCore<T>) -> (AlignfoldProbMats<T>, Sum)
where
  T: HashIndex,
{
  let (
    seq_pair,
    alignfold_scores,
    max_basepair_span_pair,
    match_probs,
    produces_struct_profs,
    trains_alignfold_scores,
    alignfold_counts_expected,
    forward_pos_pairs,
    backward_pos_pairs,
    pos_quads_hashed_lens,
    fold_scores_pair,
    produces_match_probs,
    matchable_poss,
    matchable_poss2,
  ) = inputs;
  let range_insert_scores = RangeInsertScores::new(seq_pair, alignfold_scores);
  let (alignfold_sums, global_sum) = get_alignfold_sums::<T>((
    seq_pair,
    alignfold_scores,
    max_basepair_span_pair,
    match_probs,
    trains_alignfold_scores,
    forward_pos_pairs,
    backward_pos_pairs,
    pos_quads_hashed_lens,
    fold_scores_pair,
    &range_insert_scores,
    matchable_poss,
    matchable_poss2,
  ));
  (
    get_alignfold_probs::<T>((
      seq_pair,
      alignfold_scores,
      max_basepair_span_pair,
      match_probs,
      &alignfold_sums,
      produces_struct_profs,
      global_sum,
      trains_alignfold_scores,
      alignfold_counts_expected,
      pos_quads_hashed_lens,
      fold_scores_pair,
      produces_match_probs,
      forward_pos_pairs,
      backward_pos_pairs,
      &range_insert_scores,
      matchable_poss,
      matchable_poss2,
    )),
    global_sum,
  )
}

pub fn get_alignfold_sums<T>(inputs: InputsInsideSumsGetter<T>) -> (AlignfoldSums<T>, Sum)
where
  T: HashIndex,
{
  let (
    seq_pair,
    alignfold_scores,
    max_basepair_span_pair,
    match_probs,
    trains_alignfold_scores,
    forward_pos_pairs,
    backward_pos_pairs,
    pos_quads_hashed_lens,
    fold_scores_pair,
    range_insert_scores,
    matchable_poss,
    matchable_poss2,
  ) = inputs;
  let seq_len_pair = (
    T::from_usize(seq_pair.0.len()).unwrap(),
    T::from_usize(seq_pair.1.len()).unwrap(),
  );
  let mut alignfold_sums = AlignfoldSums::<T>::new();
  for substr_len in range_inclusive(
    T::from_usize(if trains_alignfold_scores {
      2
    } else {
      MIN_SPAN_HAIRPIN_CLOSE
    })
    .unwrap(),
    max_basepair_span_pair.0,
  ) {
    for substr_len2 in range_inclusive(
      T::from_usize(if trains_alignfold_scores {
        2
      } else {
        MIN_SPAN_HAIRPIN_CLOSE
      })
      .unwrap(),
      max_basepair_span_pair.1,
    ) {
      if let Some(pos_pairs) = pos_quads_hashed_lens.get(&(substr_len, substr_len2)) {
        for &(i, k) in pos_pairs {
          let (j, l) = (i + substr_len - T::one(), k + substr_len2 - T::one());
          let (long_i, long_j, long_k, long_l) = (
            i.to_usize().unwrap(),
            j.to_usize().unwrap(),
            k.to_usize().unwrap(),
            l.to_usize().unwrap(),
          );
          let basepair = (seq_pair.0[long_i], seq_pair.0[long_j]);
          let basepair2 = (seq_pair.1[long_k], seq_pair.1[long_l]);
          let pos_quad = (i, j, k, l);
          let computes_forward_sums = true;
          let (sum_seqalign, sum_multibranch) = get_loop_sums::<T>((
            seq_pair,
            alignfold_scores,
            match_probs,
            &pos_quad,
            &mut alignfold_sums,
            computes_forward_sums,
            forward_pos_pairs,
            range_insert_scores,
            matchable_poss,
            matchable_poss2,
          ));
          let computes_forward_sums = false;
          let _ = get_loop_sums::<T>((
            seq_pair,
            alignfold_scores,
            match_probs,
            &pos_quad,
            &mut alignfold_sums,
            computes_forward_sums,
            backward_pos_pairs,
            range_insert_scores,
            matchable_poss,
            matchable_poss2,
          ));
          let mut sum = NEG_INFINITY;
          let pairmatch_score = alignfold_scores.match_scores[basepair.0][basepair2.0]
            + alignfold_scores.match_scores[basepair.1][basepair2.1];
          if substr_len.to_usize().unwrap() - 2 <= MAX_LOOP_LEN
            && substr_len2.to_usize().unwrap() - 2 <= MAX_LOOP_LEN
          {
            let hairpin_score = fold_scores_pair.0.hairpin_scores[&(i, j)];
            let hairpin_score2 = fold_scores_pair.1.hairpin_scores[&(k, l)];
            let score = hairpin_score + hairpin_score2 + sum_seqalign;
            logsumexp(&mut sum, score);
          }
          let forward_sums = &alignfold_sums.forward_sums_hashed_poss2[&(i, k)];
          let backward_sums = &alignfold_sums.backward_sums_hashed_poss2[&(j, l)];
          let min = T::from_usize(if trains_alignfold_scores {
            2
          } else {
            MIN_HAIRPIN_LEN
          })
          .unwrap();
          let min_len_pair = (
            if substr_len <= min + T::from_usize(MAX_LOOP_LEN + 2).unwrap() {
              min
            } else {
              substr_len - T::from_usize(MAX_LOOP_LEN + 2).unwrap()
            },
            if substr_len2 <= min + T::from_usize(MAX_LOOP_LEN + 2).unwrap() {
              min
            } else {
              substr_len2 - T::from_usize(MAX_LOOP_LEN + 2).unwrap()
            },
          );
          for substr_len3 in range(min_len_pair.0, substr_len - T::one()) {
            for substr_len4 in range(min_len_pair.1, substr_len2 - T::one()) {
              if let Some(pos_pairs2) = pos_quads_hashed_lens.get(&(substr_len3, substr_len4)) {
                for &(m, o) in pos_pairs2 {
                  let (n, p) = (m + substr_len3 - T::one(), o + substr_len4 - T::one());
                  if !(i < m && n < j && k < o && p < l) {
                    continue;
                  }
                  if m - i - T::one() + j - n - T::one() > T::from_usize(MAX_LOOP_LEN).unwrap() {
                    continue;
                  }
                  if o - k - T::one() + l - p - T::one() > T::from_usize(MAX_LOOP_LEN).unwrap() {
                    continue;
                  }
                  let pos_quad2 = (m, n, o, p);
                  if let Some(&x) = alignfold_sums.sums_close.get(&pos_quad2) {
                    let mut forward_term = NEG_INFINITY;
                    let mut backward_term = forward_term;
                    let pos_pair2 = (m - T::one(), o - T::one());
                    if let Some(x) = forward_sums.get(&pos_pair2) {
                      logsumexp(&mut forward_term, x.sum_seqalign);
                    }
                    let pos_pair2 = (n + T::one(), p + T::one());
                    if let Some(x) = backward_sums.get(&pos_pair2) {
                      logsumexp(&mut backward_term, x.sum_seqalign);
                    }
                    let twoloop_score = fold_scores_pair.0.twoloop_scores[&(i, j, m, n)];
                    let twoloop_score2 = fold_scores_pair.1.twoloop_scores[&(k, l, o, p)];
                    let x = twoloop_score + twoloop_score2 + x + forward_term + backward_term;
                    logsumexp(&mut sum, x);
                  }
                }
              }
            }
          }
          let multibranch_close_score = fold_scores_pair.0.multibranch_close_scores[&(i, j)];
          let multibranch_close_score2 = fold_scores_pair.1.multibranch_close_scores[&(k, l)];
          let score = multibranch_close_score + multibranch_close_score2 + sum_multibranch;
          logsumexp(&mut sum, score);
          if sum > NEG_INFINITY {
            let sum = sum + pairmatch_score;
            alignfold_sums.sums_close.insert(pos_quad, sum);
            let external_accessible_score = fold_scores_pair.0.external_accessible_scores[&(i, j)];
            let external_accessible_score2 = fold_scores_pair.1.external_accessible_scores[&(k, l)];
            alignfold_sums.sums_accessible_external.insert(
              pos_quad,
              sum + external_accessible_score + external_accessible_score2,
            );
            let multibranch_accessible_score =
              fold_scores_pair.0.multibranch_accessible_scores[&(i, j)];
            let multibranch_accessible_score2 =
              fold_scores_pair.1.multibranch_accessible_scores[&(k, l)];
            alignfold_sums.sums_accessible_multibranch.insert(
              pos_quad,
              sum + multibranch_accessible_score + multibranch_accessible_score2,
            );
          }
        }
      }
    }
  }
  let leftmost_pos_pair = (T::zero(), T::zero());
  let rightmost_pos_pair = (seq_len_pair.0 - T::one(), seq_len_pair.1 - T::one());
  alignfold_sums
    .forward_sums_external
    .insert(leftmost_pos_pair, 0.);
  alignfold_sums
    .backward_sums_external
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
      if let Some(x) = forward_pos_pairs.get(&pos_pair) {
        for &(k, l) in x {
          let pos_pair2 = (k - T::one(), l - T::one());
          let pos_quad = (k, i, l, j);
          if let Some(&x) = alignfold_sums.sums_accessible_external.get(&pos_quad) {
            if let Some(&y) = alignfold_sums.forward_sums_external2.get(&pos_pair2) {
              let y = x + y;
              logsumexp(&mut sum, y);
            }
          }
        }
      }
      let base2 = seq_pair.1[long_j];
      if i > T::zero() && j > T::zero() && match_probs.contains_key(&pos_pair) {
        let mut sum2 = NEG_INFINITY;
        let loopmatch_score =
          alignfold_scores.match_scores[base][base2] + 2. * alignfold_scores.external_score_unpair;
        let pos_pair2 = (i - T::one(), j - T::one());
        let long_pos_pair2 = (
          pos_pair2.0.to_usize().unwrap(),
          pos_pair2.1.to_usize().unwrap(),
        );
        let begins_sum = pos_pair2 == leftmost_pos_pair;
        if let Some(&x) = alignfold_sums.forward_sums_external.get(&pos_pair2) {
          let x = x
            + if begins_sum {
              alignfold_scores.init_match_score
            } else {
              alignfold_scores.match2match_score
            };
          logsumexp(&mut sum2, x);
        }
        if let Some(x) = matchable_poss.get(&pos_pair2.0) {
          for &x in x {
            if x >= pos_pair2.1 {
              continue;
            }
            let pos_pair3 = (pos_pair2.0, x);
            if let Some(&y) = alignfold_sums.forward_sums_external.get(&pos_pair3) {
              let long_x = x.to_usize().unwrap();
              let begins_sum = pos_pair3 == leftmost_pos_pair;
              let z = range_insert_scores.insert_scores_external2[long_x + 1][long_pos_pair2.1]
                + if begins_sum {
                  alignfold_scores.init_insert_score
                } else {
                  alignfold_scores.match2insert_score
                };
              let z = y + z + alignfold_scores.match2insert_score;
              logsumexp(&mut sum2, z);
            }
          }
        }
        if let Some(x) = matchable_poss2.get(&pos_pair2.1) {
          for &x in x {
            if x >= pos_pair2.0 {
              continue;
            }
            let pos_pair3 = (x, pos_pair2.1);
            if let Some(&y) = alignfold_sums.forward_sums_external.get(&pos_pair3) {
              let long_x = x.to_usize().unwrap();
              let begins_sum = pos_pair3 == leftmost_pos_pair;
              let z = range_insert_scores.insert_scores_external[long_x + 1][long_pos_pair2.0]
                + if begins_sum {
                  alignfold_scores.init_insert_score
                } else {
                  alignfold_scores.match2insert_score
                };
              let z = y + z + alignfold_scores.match2insert_score;
              logsumexp(&mut sum2, z);
            }
          }
        }
        if sum2 > NEG_INFINITY {
          alignfold_sums
            .forward_sums_external2
            .insert(pos_pair2, sum2);
        }
        let term = sum2 + loopmatch_score;
        logsumexp(&mut sum, term);
        if sum > NEG_INFINITY {
          alignfold_sums.forward_sums_external.insert(pos_pair, sum);
        }
      }
    }
  }
  let mut global_sum = NEG_INFINITY;
  let pos_pair2 = (
    rightmost_pos_pair.0 - T::one(),
    rightmost_pos_pair.1 - T::one(),
  );
  let long_pos_pair2 = (
    pos_pair2.0.to_usize().unwrap(),
    pos_pair2.1.to_usize().unwrap(),
  );
  if let Some(&x) = alignfold_sums.forward_sums_external.get(&pos_pair2) {
    logsumexp(&mut global_sum, x);
  }
  if let Some(x) = matchable_poss.get(&pos_pair2.0) {
    for &x in x {
      if x >= pos_pair2.1 {
        continue;
      }
      if let Some(&y) = alignfold_sums.forward_sums_external.get(&(pos_pair2.0, x)) {
        let long_x = x.to_usize().unwrap();
        let z = range_insert_scores.insert_scores_external2[long_x + 1][long_pos_pair2.1]
          + alignfold_scores.match2insert_score;
        let z = y + z;
        logsumexp(&mut global_sum, z);
      }
    }
  }
  if let Some(x) = matchable_poss2.get(&pos_pair2.1) {
    for &x in x {
      if x >= pos_pair2.0 {
        continue;
      }
      if let Some(&y) = alignfold_sums.forward_sums_external.get(&(x, pos_pair2.1)) {
        let long_x = x.to_usize().unwrap();
        let z = range_insert_scores.insert_scores_external[long_x + 1][long_pos_pair2.0]
          + alignfold_scores.match2insert_score;
        let z = y + z;
        logsumexp(&mut global_sum, z);
      }
    }
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
      if let Some(x) = backward_pos_pairs.get(&pos_pair) {
        for &(k, l) in x {
          let pos_pair2 = (k + T::one(), l + T::one());
          let pos_quad = (i, k, j, l);
          if let Some(&x) = alignfold_sums.sums_accessible_external.get(&pos_quad) {
            if let Some(&y) = alignfold_sums.backward_sums_external2.get(&pos_pair2) {
              let y = x + y;
              logsumexp(&mut sum, y);
            }
          }
        }
      }
      let base2 = seq_pair.1[long_j];
      if i < seq_len_pair.0 - T::one() && j < seq_len_pair.1 - T::one() {
        let pos_pair2 = (i + T::one(), j + T::one());
        let long_pos_pair2 = (
          pos_pair2.0.to_usize().unwrap(),
          pos_pair2.1.to_usize().unwrap(),
        );
        let ends_sum = pos_pair2 == rightmost_pos_pair;
        if match_probs.contains_key(&pos_pair) {
          let mut sum2 = NEG_INFINITY;
          let loopmatch_score = alignfold_scores.match_scores[base][base2]
            + 2. * alignfold_scores.external_score_unpair;
          if let Some(&x) = alignfold_sums.backward_sums_external.get(&pos_pair2) {
            let x = x
              + if ends_sum {
                0.
              } else {
                alignfold_scores.match2match_score
              };
            logsumexp(&mut sum2, x);
          }
          if let Some(x) = matchable_poss.get(&pos_pair2.0) {
            for &x in x {
              if x <= pos_pair2.1 {
                continue;
              }
              let pos_pair3 = (pos_pair2.0, x);
              if let Some(&y) = alignfold_sums.backward_sums_external.get(&pos_pair3) {
                let long_x = x.to_usize().unwrap();
                let ends_sum = pos_pair3 == rightmost_pos_pair;
                let z = range_insert_scores.insert_scores_external2[long_pos_pair2.1][long_x - 1]
                  + if ends_sum {
                    0.
                  } else {
                    alignfold_scores.match2insert_score
                  };
                let z = y + z + alignfold_scores.match2insert_score;
                logsumexp(&mut sum2, z);
              }
            }
          }
          if let Some(x) = matchable_poss2.get(&pos_pair2.1) {
            for &x in x {
              if x <= pos_pair2.0 {
                continue;
              }
              let pos_pair3 = (x, pos_pair2.1);
              if let Some(&y) = alignfold_sums.backward_sums_external.get(&pos_pair3) {
                let long_x = x.to_usize().unwrap();
                let ends_sum = pos_pair3 == rightmost_pos_pair;
                let z = range_insert_scores.insert_scores_external[long_pos_pair2.0][long_x - 1]
                  + if ends_sum {
                    0.
                  } else {
                    alignfold_scores.match2insert_score
                  };
                let z = y + z + alignfold_scores.match2insert_score;
                logsumexp(&mut sum2, z);
              }
            }
          }
          if sum2 > NEG_INFINITY {
            alignfold_sums
              .backward_sums_external2
              .insert(pos_pair2, sum2);
          }
          let term = sum2 + loopmatch_score;
          logsumexp(&mut sum, term);
          if sum > NEG_INFINITY {
            alignfold_sums.backward_sums_external.insert(pos_pair, sum);
          }
        }
      }
    }
  }
  (alignfold_sums, global_sum)
}

pub fn get_loop_sums<T>(inputs: InputsLoopSumsGetter<T>) -> (Sum, Sum)
where
  T: HashIndex,
{
  let (
    seq_pair,
    alignfold_scores,
    match_probs,
    pos_quad,
    alignfold_sums,
    computes_forward_sums,
    pos_pairs,
    range_insert_scores,
    matchable_poss,
    matchable_poss2,
  ) = inputs;
  let &(i, j, k, l) = pos_quad;
  let leftmost_pos_pair = if computes_forward_sums {
    (i, k)
  } else {
    (i + T::one(), k + T::one())
  };
  let rightmost_pos_pair = if computes_forward_sums {
    (j - T::one(), l - T::one())
  } else {
    (j, l)
  };
  let sums_hashed_poss = if computes_forward_sums {
    &mut alignfold_sums.forward_sums_hashed_poss
  } else {
    &mut alignfold_sums.backward_sums_hashed_poss
  };
  let sums_hashed_poss2 = if computes_forward_sums {
    &mut alignfold_sums.forward_sums_hashed_poss2
  } else {
    &mut alignfold_sums.backward_sums_hashed_poss2
  };
  if !sums_hashed_poss.contains_key(&if computes_forward_sums {
    leftmost_pos_pair
  } else {
    rightmost_pos_pair
  }) {
    sums_hashed_poss.insert(
      if computes_forward_sums {
        leftmost_pos_pair
      } else {
        rightmost_pos_pair
      },
      LoopSumsMat::<T>::new(),
    );
  }
  if !sums_hashed_poss2.contains_key(&if computes_forward_sums {
    leftmost_pos_pair
  } else {
    rightmost_pos_pair
  }) {
    sums_hashed_poss2.insert(
      if computes_forward_sums {
        leftmost_pos_pair
      } else {
        rightmost_pos_pair
      },
      LoopSumsMat::<T>::new(),
    );
  }
  let sums_mat = &mut sums_hashed_poss
    .get_mut(&if computes_forward_sums {
      leftmost_pos_pair
    } else {
      rightmost_pos_pair
    })
    .unwrap();
  let sums_mat2 = &mut sums_hashed_poss2
    .get_mut(&if computes_forward_sums {
      leftmost_pos_pair
    } else {
      rightmost_pos_pair
    })
    .unwrap();
  let iter: Poss<T> = if computes_forward_sums {
    range(i, j).collect()
  } else {
    range_inclusive(i + T::one(), j).rev().collect()
  };
  let iter2: Poss<T> = if computes_forward_sums {
    range(k, l).collect()
  } else {
    range_inclusive(k + T::one(), l).rev().collect()
  };
  for &u in iter.iter() {
    let long_u = u.to_usize().unwrap();
    let base = seq_pair.0[long_u];
    for &v in iter2.iter() {
      let pos_pair = (u, v);
      if sums_mat.contains_key(&pos_pair) {
        continue;
      }
      let mut sums = LoopSums::new();
      if (computes_forward_sums && u == i && v == k) || (!computes_forward_sums && u == j && v == l)
      {
        sums.sum_seqalign = 0.;
        sums.sum_seqalign_multibranch = 0.;
        sums.sum_0ormore_pairmatches = 0.;
        sums_mat.insert(pos_pair, sums);
        continue;
      }
      let long_v = v.to_usize().unwrap();
      let mut sum_multibranch = NEG_INFINITY;
      let mut sum_1st_pairmatches = sum_multibranch;
      let mut sum = sum_multibranch;
      if let Some(pos_pairs) = pos_pairs.get(&pos_pair) {
        for &(m, n) in pos_pairs {
          if computes_forward_sums {
            if !(i < m && k < n) {
              continue;
            }
          } else if !(m < j && n < l) {
            continue;
          }
          let pos_pair2 = if computes_forward_sums {
            (m - T::one(), n - T::one())
          } else {
            (m + T::one(), n + T::one())
          };
          let pos_quad2 = if computes_forward_sums {
            (m, u, n, v)
          } else {
            (u, m, v, n)
          };
          if let Some(&x) = alignfold_sums.sums_accessible_multibranch.get(&pos_quad2) {
            if let Some(y) = sums_mat2.get(&pos_pair2) {
              let z = x + y.sum_1ormore_pairmatches;
              logsumexp(&mut sum_multibranch, z);
              let z = x + y.sum_seqalign_multibranch;
              logsumexp(&mut sum_1st_pairmatches, z);
            }
          }
        }
      }
      let pos_pair2 = if computes_forward_sums {
        (u - T::one(), v - T::one())
      } else {
        (u + T::one(), v + T::one())
      };
      let long_pos_pair2 = (
        pos_pair2.0.to_usize().unwrap(),
        pos_pair2.1.to_usize().unwrap(),
      );
      let base2 = seq_pair.1[long_v];
      if match_probs.contains_key(&pos_pair) {
        let mut sums2 = LoopSums::new();
        let mut sum_seqalign2 = NEG_INFINITY;
        let mut sum_seqalign_multibranch2 = sum_seqalign2;
        let mut sum_multibranch2 = sum_seqalign2;
        let mut sum_1st_pairmatches2 = sum_seqalign2;
        let mut sum2 = sum_seqalign2;
        let loopmatch_score = alignfold_scores.match_scores[base][base2];
        let loopmatch_score_multibranch =
          loopmatch_score + 2. * alignfold_scores.multibranch_score_unpair;
        if let Some(x) = sums_mat.get(&pos_pair2) {
          let y = x.sum_multibranch + alignfold_scores.match2match_score;
          logsumexp(&mut sum_multibranch2, y);
          let y = x.sum_1st_pairmatches + alignfold_scores.match2match_score;
          logsumexp(&mut sum_1st_pairmatches2, y);
          let y = x.sum_seqalign_multibranch + alignfold_scores.match2match_score;
          logsumexp(&mut sum_seqalign_multibranch2, y);
          let y = x.sum_seqalign + alignfold_scores.match2match_score;
          logsumexp(&mut sum_seqalign2, y);
        }
        if let Some(x) = matchable_poss.get(&pos_pair2.0) {
          for &x in x {
            if computes_forward_sums && x >= pos_pair2.1
              || (!computes_forward_sums && x <= pos_pair2.1)
            {
              continue;
            }
            if let Some(y) = sums_mat.get(&(pos_pair2.0, x)) {
              let long_x = x.to_usize().unwrap();
              let z = if computes_forward_sums {
                range_insert_scores.insert_scores2[long_x + 1][long_pos_pair2.1]
              } else {
                range_insert_scores.insert_scores2[long_pos_pair2.1][long_x - 1]
              } + alignfold_scores.match2insert_score;
              let a = if computes_forward_sums {
                range_insert_scores.insert_scores_multibranch2[long_x + 1][long_pos_pair2.1]
              } else {
                range_insert_scores.insert_scores_multibranch2[long_pos_pair2.1][long_x - 1]
              } + alignfold_scores.match2insert_score;
              let x = y.sum_multibranch + alignfold_scores.match2insert_score + a;
              logsumexp(&mut sum_multibranch2, x);
              let x = y.sum_1st_pairmatches + alignfold_scores.match2insert_score + a;
              logsumexp(&mut sum_1st_pairmatches2, x);
              let x = y.sum_seqalign + alignfold_scores.match2insert_score + z;
              logsumexp(&mut sum_seqalign2, x);
              let x = y.sum_seqalign_multibranch + alignfold_scores.match2insert_score + a;
              logsumexp(&mut sum_seqalign_multibranch2, x);
            }
          }
        }
        if let Some(x) = matchable_poss2.get(&pos_pair2.1) {
          for &x in x {
            if computes_forward_sums && x >= pos_pair2.0
              || (!computes_forward_sums && x <= pos_pair2.0)
            {
              continue;
            }
            if let Some(y) = sums_mat.get(&(x, pos_pair2.1)) {
              let long_x = x.to_usize().unwrap();
              let z = if computes_forward_sums {
                range_insert_scores.insert_scores[long_x + 1][long_pos_pair2.0]
              } else {
                range_insert_scores.insert_scores[long_pos_pair2.0][long_x - 1]
              } + alignfold_scores.match2insert_score;
              let a = if computes_forward_sums {
                range_insert_scores.insert_scores_multibranch[long_x + 1][long_pos_pair2.0]
              } else {
                range_insert_scores.insert_scores_multibranch[long_pos_pair2.0][long_x - 1]
              } + alignfold_scores.match2insert_score;
              let x = y.sum_multibranch + alignfold_scores.match2insert_score + a;
              logsumexp(&mut sum_multibranch2, x);
              let x = y.sum_1st_pairmatches + alignfold_scores.match2insert_score + a;
              logsumexp(&mut sum_1st_pairmatches2, x);
              let x = y.sum_seqalign + alignfold_scores.match2insert_score + z;
              logsumexp(&mut sum_seqalign2, x);
              let x = y.sum_seqalign_multibranch + alignfold_scores.match2insert_score + a;
              logsumexp(&mut sum_seqalign_multibranch2, x);
            }
          }
        }
        sums2.sum_multibranch = sum_multibranch2;
        logsumexp(&mut sum2, sum_multibranch2);
        sums2.sum_1st_pairmatches = sum_1st_pairmatches2;
        logsumexp(&mut sum2, sum_1st_pairmatches2);
        sums2.sum_1ormore_pairmatches = sum2;
        sums2.sum_seqalign = sum_seqalign2;
        sums2.sum_seqalign_multibranch = sum_seqalign_multibranch2;
        logsumexp(&mut sum2, sum_seqalign_multibranch2);
        sums2.sum_0ormore_pairmatches = sum2;
        if has_valid_sums(&sums2) {
          sums_mat2.insert(pos_pair2, sums2);
        }
        let term = sum_multibranch2 + loopmatch_score_multibranch;
        logsumexp(&mut sum_multibranch, term);
        sums.sum_multibranch = sum_multibranch;
        logsumexp(&mut sum, sum_multibranch);
        let term = sum_1st_pairmatches2 + loopmatch_score_multibranch;
        logsumexp(&mut sum_1st_pairmatches, term);
        sums.sum_1st_pairmatches = sum_1st_pairmatches;
        logsumexp(&mut sum, sum_1st_pairmatches);
        sums.sum_1ormore_pairmatches = sum;
        let sum_seqalign_multibranch = sum_seqalign_multibranch2 + loopmatch_score_multibranch;
        sums.sum_seqalign_multibranch = sum_seqalign_multibranch;
        logsumexp(&mut sum, sum_seqalign_multibranch);
        sums.sum_0ormore_pairmatches = sum;
        let sum_seqalign = sum_seqalign2 + loopmatch_score;
        sums.sum_seqalign = sum_seqalign;
        if has_valid_sums(&sums) {
          sums_mat.insert(pos_pair, sums);
        }
      }
    }
  }
  let mut final_sum_seqalign = NEG_INFINITY;
  let mut final_sum_multibranch = final_sum_seqalign;
  if computes_forward_sums {
    let pos_pair2 = rightmost_pos_pair;
    let long_pos_pair2 = (
      pos_pair2.0.to_usize().unwrap(),
      pos_pair2.1.to_usize().unwrap(),
    );
    if let Some(x) = sums_mat.get(&pos_pair2) {
      let y = x.sum_multibranch + alignfold_scores.match2match_score;
      logsumexp(&mut final_sum_multibranch, y);
      let y = x.sum_seqalign + alignfold_scores.match2match_score;
      logsumexp(&mut final_sum_seqalign, y);
    }
    if let Some(x) = matchable_poss.get(&pos_pair2.0) {
      for &x in x {
        if x >= pos_pair2.1 {
          continue;
        }
        if let Some(y) = sums_mat.get(&(pos_pair2.0, x)) {
          let long_x = x.to_usize().unwrap();
          let z = range_insert_scores.insert_scores2[long_x + 1][long_pos_pair2.1]
            + alignfold_scores.match2insert_score;
          let a = range_insert_scores.insert_scores_multibranch2[long_x + 1][long_pos_pair2.1]
            + alignfold_scores.match2insert_score;
          let x = y.sum_multibranch + alignfold_scores.match2insert_score + a;
          logsumexp(&mut final_sum_multibranch, x);
          let x = y.sum_seqalign + alignfold_scores.match2insert_score + z;
          logsumexp(&mut final_sum_seqalign, x);
        }
      }
    }
    if let Some(x) = matchable_poss2.get(&pos_pair2.1) {
      for &x in x {
        if x >= pos_pair2.0 {
          continue;
        }
        if let Some(y) = sums_mat.get(&(x, pos_pair2.1)) {
          let long_x = x.to_usize().unwrap();
          let z = range_insert_scores.insert_scores[long_x + 1][long_pos_pair2.0]
            + alignfold_scores.match2insert_score;
          let a = range_insert_scores.insert_scores_multibranch[long_x + 1][long_pos_pair2.0]
            + alignfold_scores.match2insert_score;
          let x = y.sum_multibranch + alignfold_scores.match2insert_score + a;
          logsumexp(&mut final_sum_multibranch, x);
          let x = y.sum_seqalign + alignfold_scores.match2insert_score + z;
          logsumexp(&mut final_sum_seqalign, x);
        }
      }
    }
    let mut sums2 = LoopSums::new();
    sums2.sum_multibranch = final_sum_multibranch;
    sums2.sum_seqalign = final_sum_seqalign;
    sums_mat2.insert(pos_pair2, sums2);
  }
  (final_sum_seqalign, final_sum_multibranch)
}

pub fn get_2loop_sums<T>(inputs: Inputs2loopSumsGetter<T>) -> (SparseSumMat<T>, SparseSumMat<T>)
where
  T: HashIndex,
{
  let (
    seq_pair,
    alignfold_scores,
    match_probs,
    pos_quad,
    alignfold_sums,
    computes_forward_sums,
    pos_pairs,
    fold_scores_pair,
    range_insert_scores,
    matchable_poss,
    matchable_poss2,
  ) = inputs;
  let &(i, j, k, l) = pos_quad;
  let leftmost_pos_pair = if computes_forward_sums {
    (i, k)
  } else {
    (i + T::one(), k + T::one())
  };
  let rightmost_pos_pair = if computes_forward_sums {
    (j - T::one(), l - T::one())
  } else {
    (j, l)
  };
  let sums_hashed_poss = if computes_forward_sums {
    &alignfold_sums.forward_sums_hashed_poss2
  } else {
    &alignfold_sums.backward_sums_hashed_poss2
  };
  let sums = &sums_hashed_poss[&if computes_forward_sums {
    leftmost_pos_pair
  } else {
    rightmost_pos_pair
  }];
  let iter: Poss<T> = if computes_forward_sums {
    range(i, j).collect()
  } else {
    range_inclusive(i + T::one(), j).rev().collect()
  };
  let iter2: Poss<T> = if computes_forward_sums {
    range(k, l).collect()
  } else {
    range_inclusive(k + T::one(), l).rev().collect()
  };
  let mut sum_mat = SparseSumMat::<T>::default();
  let mut sum_mat2 = sum_mat.clone();
  for &u in iter.iter() {
    let long_u = u.to_usize().unwrap();
    let base = seq_pair.0[long_u];
    for &v in iter2.iter() {
      let pos_pair = (u, v);
      if (computes_forward_sums && u == i && v == k) || (!computes_forward_sums && u == j && v == l)
      {
        continue;
      }
      let long_v = v.to_usize().unwrap();
      let mut sum = NEG_INFINITY;
      if let Some(x) = pos_pairs.get(&pos_pair) {
        for &(m, n) in x {
          if computes_forward_sums {
            if !(i < m && k < n) {
              continue;
            }
          } else if !(m < j && n < l) {
            continue;
          }
          let pos_pair2 = if computes_forward_sums {
            (m - T::one(), n - T::one())
          } else {
            (m + T::one(), n + T::one())
          };
          let pos_quad2 = if computes_forward_sums {
            (m, u, n, v)
          } else {
            (u, m, v, n)
          };
          if pos_quad2.0 - i - T::one() + j - pos_quad2.1 - T::one()
            > T::from_usize(MAX_LOOP_LEN).unwrap()
          {
            continue;
          }
          if pos_quad2.2 - k - T::one() + l - pos_quad2.3 - T::one()
            > T::from_usize(MAX_LOOP_LEN).unwrap()
          {
            continue;
          }
          if let Some(&x) = alignfold_sums.sums_close.get(&pos_quad2) {
            if let Some(y) = sums.get(&pos_pair2) {
              let twoloop_score =
                fold_scores_pair.0.twoloop_scores[&(i, j, pos_quad2.0, pos_quad2.1)];
              let twoloop_score2 =
                fold_scores_pair.1.twoloop_scores[&(k, l, pos_quad2.2, pos_quad2.3)];
              let y = x + y.sum_seqalign + twoloop_score + twoloop_score2;
              logsumexp(&mut sum, y);
            }
          }
        }
      }
      let pos_pair2 = if computes_forward_sums {
        (u - T::one(), v - T::one())
      } else {
        (u + T::one(), v + T::one())
      };
      let long_pos_pair2 = (
        pos_pair2.0.to_usize().unwrap(),
        pos_pair2.1.to_usize().unwrap(),
      );
      let base2 = seq_pair.1[long_v];
      if match_probs.contains_key(&pos_pair) {
        let mut sum2 = NEG_INFINITY;
        let loopmatch_score = alignfold_scores.match_scores[base][base2];
        if let Some(&x) = sum_mat.get(&pos_pair2) {
          let x = x + alignfold_scores.match2match_score;
          logsumexp(&mut sum2, x);
        }
        if let Some(x) = matchable_poss.get(&pos_pair2.0) {
          for &x in x {
            if computes_forward_sums && x >= pos_pair2.1
              || (!computes_forward_sums && x <= pos_pair2.1)
            {
              continue;
            }
            if let Some(&y) = sum_mat.get(&(pos_pair2.0, x)) {
              let long_x = x.to_usize().unwrap();
              let z = if computes_forward_sums {
                range_insert_scores.insert_scores2[long_x + 1][long_pos_pair2.1]
              } else {
                range_insert_scores.insert_scores2[long_pos_pair2.1][long_x - 1]
              } + alignfold_scores.match2insert_score;
              let z = y + alignfold_scores.match2insert_score + z;
              logsumexp(&mut sum2, z);
            }
          }
        }
        if let Some(x) = matchable_poss2.get(&pos_pair2.1) {
          for &x in x {
            if computes_forward_sums && x >= pos_pair2.0
              || (!computes_forward_sums && x <= pos_pair2.0)
            {
              continue;
            }
            if let Some(&y) = sum_mat.get(&(x, pos_pair2.1)) {
              let long_x = x.to_usize().unwrap();
              let z = if computes_forward_sums {
                range_insert_scores.insert_scores[long_x + 1][long_pos_pair2.0]
              } else {
                range_insert_scores.insert_scores[long_pos_pair2.0][long_x - 1]
              } + alignfold_scores.match2insert_score;
              let z = y + alignfold_scores.match2insert_score + z;
              logsumexp(&mut sum2, z);
            }
          }
        }
        if sum2 > NEG_INFINITY {
          sum_mat2.insert(pos_pair2, sum2);
        }
        let term = sum2 + loopmatch_score;
        logsumexp(&mut sum, term);
        if sum > NEG_INFINITY {
          sum_mat.insert(pos_pair, sum);
        }
      }
    }
  }
  (sum_mat, sum_mat2)
}

pub fn get_alignfold_probs<T>(inputs: InputsAlignfoldProbsGetter<T>) -> AlignfoldProbMats<T>
where
  T: HashIndex,
{
  let (
    seq_pair,
    alignfold_scores,
    max_basepair_span_pair,
    match_probs,
    alignfold_sums,
    produces_struct_profs,
    global_sum,
    trains_alignfold_scores,
    alignfold_counts_expected,
    pos_quads_hashed_lens,
    fold_scores_pair,
    produces_match_probs,
    forward_pos_pairs,
    backward_pos_pairs,
    range_insert_scores,
    matchable_poss,
    matchable_poss2,
  ) = inputs;
  let seq_len_pair = (
    T::from_usize(seq_pair.0.len()).unwrap(),
    T::from_usize(seq_pair.1.len()).unwrap(),
  );
  let mut alignfold_outside_sums = SumMat4d::<T>::default();
  let mut alignfold_probs = AlignfoldProbMats::<T>::new(&(
    seq_len_pair.0.to_usize().unwrap(),
    seq_len_pair.1.to_usize().unwrap(),
  ));
  let leftmost_pos_pair = (T::zero(), T::zero());
  let rightmost_pos_pair = (seq_len_pair.0 - T::one(), seq_len_pair.1 - T::one());
  let mut prob_coeffs_multibranch = SumMat4d::<T>::default();
  let mut prob_coeffs_multibranch2 = prob_coeffs_multibranch.clone();
  for substr_len in range_inclusive(
    T::from_usize(if trains_alignfold_scores {
      2
    } else {
      MIN_SPAN_HAIRPIN_CLOSE
    })
    .unwrap(),
    max_basepair_span_pair.0,
  )
  .rev()
  {
    for substr_len2 in range_inclusive(
      T::from_usize(if trains_alignfold_scores {
        2
      } else {
        MIN_SPAN_HAIRPIN_CLOSE
      })
      .unwrap(),
      max_basepair_span_pair.1,
    )
    .rev()
    {
      if let Some(pos_pairs) = pos_quads_hashed_lens.get(&(substr_len, substr_len2)) {
        for &(i, k) in pos_pairs {
          let (j, l) = (i + substr_len - T::one(), k + substr_len2 - T::one());
          let pos_quad = (i, j, k, l);
          if let Some(&sum_close) = alignfold_sums.sums_close.get(&pos_quad) {
            let (long_i, long_j, long_k, long_l) = (
              i.to_usize().unwrap(),
              j.to_usize().unwrap(),
              k.to_usize().unwrap(),
              l.to_usize().unwrap(),
            );
            let basepair = (seq_pair.0[long_i], seq_pair.0[long_j]);
            let basepair2 = (seq_pair.1[long_k], seq_pair.1[long_l]);
            let mismatch_pair = (seq_pair.0[long_i - 1], seq_pair.0[long_j + 1]);
            let mismatch_pair2 = (seq_pair.1[long_k - 1], seq_pair.1[long_l + 1]);
            let prob_coeff = sum_close - global_sum;
            let mut sum = NEG_INFINITY;
            let mut forward_term = sum;
            let mut forward_term_match = sum;
            let mut backward_term = sum;
            let pos_pair2 = (i - T::one(), k - T::one());
            if let Some(&x) = alignfold_sums.forward_sums_external2.get(&pos_pair2) {
              logsumexp(&mut forward_term, x);
            }
            if trains_alignfold_scores {
              if let Some(&x) = alignfold_sums.forward_sums_external.get(&pos_pair2) {
                let begins_sum = pos_pair2 == leftmost_pos_pair;
                let x = x
                  + if begins_sum {
                    alignfold_scores.init_match_score
                  } else {
                    alignfold_scores.match2match_score
                  };
                logsumexp(&mut forward_term_match, x);
              }
            }
            let pos_pair2 = (j + T::one(), l + T::one());
            if let Some(&x) = alignfold_sums.backward_sums_external2.get(&pos_pair2) {
              logsumexp(&mut backward_term, x);
            }
            let coeff = alignfold_sums.sums_accessible_external[&pos_quad] - sum_close;
            if trains_alignfold_scores {
              let begins_sum = (i - T::one(), k - T::one()) == leftmost_pos_pair;
              let x = prob_coeff + coeff + forward_term_match + backward_term;
              if begins_sum {
                logsumexp(&mut alignfold_counts_expected.init_match_score, x);
              } else {
                logsumexp(&mut alignfold_counts_expected.match2match_score, x);
              }
            }
            let sum_external = forward_term + backward_term;
            if sum_external > NEG_INFINITY {
              sum = coeff + sum_external;
              let x = prob_coeff + sum;
              if trains_alignfold_scores {
                // Count external loop accessible basepairings.
                logsumexp(
                  &mut alignfold_counts_expected.external_score_basepair,
                  (2. as Prob).ln() + x,
                );
                // Count helix ends.
                logsumexp(
                  &mut alignfold_counts_expected.helix_close_scores[basepair.1][basepair.0],
                  x,
                );
                logsumexp(
                  &mut alignfold_counts_expected.helix_close_scores[basepair2.1][basepair2.0],
                  x,
                );
                // Count external loop terminal mismatches.
                if j < seq_len_pair.0 - T::from_usize(2).unwrap() {
                  logsumexp(
                    &mut alignfold_counts_expected.dangling_scores_left[basepair.1][basepair.0]
                      [mismatch_pair.1],
                    x,
                  );
                }
                if i > T::one() {
                  logsumexp(
                    &mut alignfold_counts_expected.dangling_scores_right[basepair.1][basepair.0]
                      [mismatch_pair.0],
                    x,
                  );
                }
                if l < seq_len_pair.1 - T::from_usize(2).unwrap() {
                  logsumexp(
                    &mut alignfold_counts_expected.dangling_scores_left[basepair2.1][basepair2.0]
                      [mismatch_pair2.1],
                    x,
                  );
                }
                if k > T::one() {
                  logsumexp(
                    &mut alignfold_counts_expected.dangling_scores_right[basepair2.1][basepair2.0]
                      [mismatch_pair2.0],
                    x,
                  );
                }
              }
            }
            for substr_len3 in range_inclusive(
              substr_len + T::from_usize(2).unwrap(),
              (substr_len + T::from_usize(MAX_LOOP_LEN + 2).unwrap()).min(max_basepair_span_pair.0),
            ) {
              for substr_len4 in range_inclusive(
                substr_len2 + T::from_usize(2).unwrap(),
                (substr_len2 + T::from_usize(MAX_LOOP_LEN + 2).unwrap())
                  .min(max_basepair_span_pair.1),
              ) {
                if let Some(pos_pairs2) = pos_quads_hashed_lens.get(&(substr_len3, substr_len4)) {
                  for &(m, o) in pos_pairs2 {
                    let (n, p) = (m + substr_len3 - T::one(), o + substr_len4 - T::one());
                    if !(m < i && j < n && o < k && l < p) {
                      continue;
                    }
                    let (long_m, long_n, long_o, long_p) = (
                      m.to_usize().unwrap(),
                      n.to_usize().unwrap(),
                      o.to_usize().unwrap(),
                      p.to_usize().unwrap(),
                    );
                    let loop_len_pair = (long_i - long_m - 1, long_n - long_j - 1);
                    let loop_len_pair2 = (long_k - long_o - 1, long_p - long_l - 1);
                    if loop_len_pair.0 + loop_len_pair.1 > MAX_LOOP_LEN {
                      continue;
                    }
                    if loop_len_pair2.0 + loop_len_pair2.1 > MAX_LOOP_LEN {
                      continue;
                    }
                    let basepair3 = (seq_pair.0[long_m], seq_pair.0[long_n]);
                    let basepair4 = (seq_pair.1[long_o], seq_pair.1[long_p]);
                    let found_stack = loop_len_pair.0 == 0 && loop_len_pair.1 == 0;
                    let found_bulge =
                      (loop_len_pair.0 == 0 || loop_len_pair.1 == 0) && !found_stack;
                    let mismatch_pair3 = (seq_pair.0[long_m + 1], seq_pair.0[long_n - 1]);
                    let pos_quad2 = (m, n, o, p);
                    if let Some(&outside_sum) = alignfold_outside_sums.get(&pos_quad2) {
                      let found_stack2 = loop_len_pair2.0 == 0 && loop_len_pair2.1 == 0;
                      let found_bulge2 =
                        (loop_len_pair2.0 == 0 || loop_len_pair2.1 == 0) && !found_stack2;
                      let mismatch_pair4 = (seq_pair.1[long_o + 1], seq_pair.1[long_p - 1]);
                      let forward_sums = &alignfold_sums.forward_sums_hashed_poss[&(m, o)];
                      let forward_sums2 = &alignfold_sums.forward_sums_hashed_poss2[&(m, o)];
                      let backward_sums = &alignfold_sums.backward_sums_hashed_poss2[&(n, p)];
                      let mut forward_term = NEG_INFINITY;
                      let mut forward_term_match = forward_term;
                      let mut backward_term = forward_term;
                      let pos_pair2 = (i - T::one(), k - T::one());
                      if let Some(x) = forward_sums2.get(&pos_pair2) {
                        logsumexp(&mut forward_term, x.sum_seqalign);
                      }
                      if trains_alignfold_scores {
                        if let Some(x) = forward_sums.get(&pos_pair2) {
                          let term = x.sum_seqalign + alignfold_scores.match2match_score;
                          logsumexp(&mut forward_term_match, term);
                        }
                      }
                      let pos_pair2 = (j + T::one(), l + T::one());
                      if let Some(x) = backward_sums.get(&pos_pair2) {
                        logsumexp(&mut backward_term, x.sum_seqalign);
                      }
                      let pairmatch_score = alignfold_scores.match_scores[basepair3.0][basepair4.0]
                        + alignfold_scores.match_scores[basepair3.1][basepair4.1];
                      let twoloop_score = fold_scores_pair.0.twoloop_scores[&(m, n, i, j)];
                      let twoloop_score2 = fold_scores_pair.1.twoloop_scores[&(o, p, k, l)];
                      let coeff = pairmatch_score + twoloop_score + twoloop_score2 + outside_sum;
                      if trains_alignfold_scores {
                        let x = prob_coeff + coeff + forward_term_match + backward_term;
                        logsumexp(&mut alignfold_counts_expected.match2match_score, x);
                      }
                      let sum_2loop = forward_term + backward_term;
                      if sum_2loop > NEG_INFINITY {
                        let sum_2loop = coeff + sum_2loop;
                        logsumexp(&mut sum, sum_2loop);
                        let pairmatch_prob_2loop = prob_coeff + sum_2loop;
                        if produces_struct_profs {
                          let loop_len_pair = (long_i - long_m - 1, long_n - long_j - 1);
                          let found_bulge = (loop_len_pair.0 == 0) ^ (loop_len_pair.1 == 0);
                          let found_interior = loop_len_pair.0 > 0 && loop_len_pair.1 > 0;
                          for q in long_m + 1..long_i {
                            if found_bulge {
                              logsumexp(
                                &mut alignfold_probs.context_profs_pair.0[(q, CONTEXT_INDEX_BULGE)],
                                pairmatch_prob_2loop,
                              );
                            } else if found_interior {
                              logsumexp(
                                &mut alignfold_probs.context_profs_pair.0
                                  [(q, CONTEXT_INDEX_INTERIOR)],
                                pairmatch_prob_2loop,
                              );
                            }
                          }
                          for q in long_j + 1..long_n {
                            if found_bulge {
                              logsumexp(
                                &mut alignfold_probs.context_profs_pair.0[(q, CONTEXT_INDEX_BULGE)],
                                pairmatch_prob_2loop,
                              );
                            } else if found_interior {
                              logsumexp(
                                &mut alignfold_probs.context_profs_pair.0
                                  [(q, CONTEXT_INDEX_INTERIOR)],
                                pairmatch_prob_2loop,
                              );
                            }
                          }
                          let loop_len_pair = (long_k - long_o - 1, long_p - long_l - 1);
                          let found_bulge = (loop_len_pair.0 == 0) ^ (loop_len_pair.1 == 0);
                          let found_interior = loop_len_pair.0 > 0 && loop_len_pair.1 > 0;
                          for q in long_o + 1..long_k {
                            if found_bulge {
                              logsumexp(
                                &mut alignfold_probs.context_profs_pair.1[(q, CONTEXT_INDEX_BULGE)],
                                pairmatch_prob_2loop,
                              );
                            } else if found_interior {
                              logsumexp(
                                &mut alignfold_probs.context_profs_pair.1
                                  [(q, CONTEXT_INDEX_INTERIOR)],
                                pairmatch_prob_2loop,
                              );
                            }
                          }
                          for q in long_l + 1..long_p {
                            if found_bulge {
                              logsumexp(
                                &mut alignfold_probs.context_profs_pair.1[(q, CONTEXT_INDEX_BULGE)],
                                pairmatch_prob_2loop,
                              );
                            } else if found_interior {
                              logsumexp(
                                &mut alignfold_probs.context_profs_pair.1
                                  [(q, CONTEXT_INDEX_INTERIOR)],
                                pairmatch_prob_2loop,
                              );
                            }
                          }
                        }
                        if trains_alignfold_scores {
                          if found_stack {
                            // Count a stack.
                            let dict_min_stack = get_dict_min_stack(&basepair3, &basepair);
                            logsumexp(
                              &mut alignfold_counts_expected.stack_scores[dict_min_stack.0 .0]
                                [dict_min_stack.0 .1][dict_min_stack.1 .0][dict_min_stack.1 .1],
                              pairmatch_prob_2loop,
                            );
                          } else {
                            if found_bulge {
                              // Count a bulge loop length.
                              let bulge_len = if loop_len_pair.0 == 0 {
                                loop_len_pair.1
                              } else {
                                loop_len_pair.0
                              };
                              logsumexp(
                                &mut alignfold_counts_expected.bulge_scores_len[bulge_len - 1],
                                pairmatch_prob_2loop,
                              );
                              // Count a 0x1 bulge loop.
                              if bulge_len == 1 {
                                let mismatch = if loop_len_pair.0 == 0 {
                                  mismatch_pair3.1
                                } else {
                                  mismatch_pair3.0
                                };
                                logsumexp(
                                  &mut alignfold_counts_expected.bulge_scores_0x1[mismatch],
                                  pairmatch_prob_2loop,
                                );
                              }
                            } else {
                              // Count an interior loop length.
                              logsumexp(
                                &mut alignfold_counts_expected.interior_scores_len
                                  [loop_len_pair.0 + loop_len_pair.1 - 2],
                                pairmatch_prob_2loop,
                              );
                              let diff = get_diff(loop_len_pair.0, loop_len_pair.1);
                              if diff == 0 {
                                logsumexp(
                                  &mut alignfold_counts_expected.interior_scores_symmetric
                                    [loop_len_pair.0 - 1],
                                  pairmatch_prob_2loop,
                                );
                              } else {
                                logsumexp(
                                  &mut alignfold_counts_expected.interior_scores_asymmetric
                                    [diff - 1],
                                  pairmatch_prob_2loop,
                                );
                              }
                              // Count a 1x1 interior loop.
                              if loop_len_pair.0 == 1 && loop_len_pair.1 == 1 {
                                let dict_min_mismatch_pair3 = get_dict_min_pair(&mismatch_pair3);
                                logsumexp(
                                  &mut alignfold_counts_expected.interior_scores_1x1
                                    [dict_min_mismatch_pair3.0][dict_min_mismatch_pair3.1],
                                  pairmatch_prob_2loop,
                                );
                              }
                              // Count an explicit interior loop length pair.
                              if loop_len_pair.0 <= MAX_INTERIOR_EXPLICIT
                                && loop_len_pair.1 <= MAX_INTERIOR_EXPLICIT
                              {
                                let dict_min_len_pair = get_dict_min_pair(&loop_len_pair);
                                logsumexp(
                                  &mut alignfold_counts_expected.interior_scores_explicit
                                    [dict_min_len_pair.0 - 1][dict_min_len_pair.1 - 1],
                                  pairmatch_prob_2loop,
                                );
                              }
                            }
                            // Count helix ends.
                            logsumexp(
                              &mut alignfold_counts_expected.helix_close_scores[basepair.1]
                                [basepair.0],
                              pairmatch_prob_2loop,
                            );
                            logsumexp(
                              &mut alignfold_counts_expected.helix_close_scores[basepair3.0]
                                [basepair3.1],
                              pairmatch_prob_2loop,
                            );
                            // Count 2-loop terminal mismatches.
                            logsumexp(
                              &mut alignfold_counts_expected.terminal_mismatch_scores[basepair3.0]
                                [basepair3.1][mismatch_pair3.0][mismatch_pair3.1],
                              pairmatch_prob_2loop,
                            );
                            logsumexp(
                              &mut alignfold_counts_expected.terminal_mismatch_scores[basepair.1]
                                [basepair.0][mismatch_pair.1][mismatch_pair.0],
                              pairmatch_prob_2loop,
                            );
                          }
                          if found_stack2 {
                            // Count a stack.
                            let dict_min_stack2 = get_dict_min_stack(&basepair4, &basepair2);
                            logsumexp(
                              &mut alignfold_counts_expected.stack_scores[dict_min_stack2.0 .0]
                                [dict_min_stack2.0 .1][dict_min_stack2.1 .0][dict_min_stack2.1 .1],
                              pairmatch_prob_2loop,
                            );
                          } else {
                            if found_bulge2 {
                              // Count a bulge loop length.
                              let bulge_len2 = if loop_len_pair2.0 == 0 {
                                loop_len_pair2.1
                              } else {
                                loop_len_pair2.0
                              };
                              logsumexp(
                                &mut alignfold_counts_expected.bulge_scores_len[bulge_len2 - 1],
                                pairmatch_prob_2loop,
                              );
                              // Count a 0x1 bulge loop.
                              if bulge_len2 == 1 {
                                let mismatch2 = if loop_len_pair2.0 == 0 {
                                  mismatch_pair4.1
                                } else {
                                  mismatch_pair4.0
                                };
                                logsumexp(
                                  &mut alignfold_counts_expected.bulge_scores_0x1[mismatch2],
                                  pairmatch_prob_2loop,
                                );
                              }
                            } else {
                              // Count an interior loop length.
                              logsumexp(
                                &mut alignfold_counts_expected.interior_scores_len
                                  [loop_len_pair2.0 + loop_len_pair2.1 - 2],
                                pairmatch_prob_2loop,
                              );
                              let diff2 = get_diff(loop_len_pair2.0, loop_len_pair2.1);
                              if diff2 == 0 {
                                logsumexp(
                                  &mut alignfold_counts_expected.interior_scores_symmetric
                                    [loop_len_pair2.0 - 1],
                                  pairmatch_prob_2loop,
                                );
                              } else {
                                logsumexp(
                                  &mut alignfold_counts_expected.interior_scores_asymmetric
                                    [diff2 - 1],
                                  pairmatch_prob_2loop,
                                );
                              }
                              // Count a 1x1 interior loop.
                              if loop_len_pair2.0 == 1 && loop_len_pair2.1 == 1 {
                                let dict_min_mismatch_pair4 = get_dict_min_pair(&mismatch_pair4);
                                logsumexp(
                                  &mut alignfold_counts_expected.interior_scores_1x1
                                    [dict_min_mismatch_pair4.0][dict_min_mismatch_pair4.1],
                                  pairmatch_prob_2loop,
                                );
                              }
                              // Count an explicit interior loop length pair.
                              if loop_len_pair2.0 <= MAX_INTERIOR_EXPLICIT
                                && loop_len_pair2.1 <= MAX_INTERIOR_EXPLICIT
                              {
                                let dict_min_len_pair2 = get_dict_min_pair(&loop_len_pair2);
                                logsumexp(
                                  &mut alignfold_counts_expected.interior_scores_explicit
                                    [dict_min_len_pair2.0 - 1][dict_min_len_pair2.1 - 1],
                                  pairmatch_prob_2loop,
                                );
                              }
                            }
                            // Count helix ends.
                            logsumexp(
                              &mut alignfold_counts_expected.helix_close_scores[basepair2.1]
                                [basepair2.0],
                              pairmatch_prob_2loop,
                            );
                            logsumexp(
                              &mut alignfold_counts_expected.helix_close_scores[basepair4.0]
                                [basepair4.1],
                              pairmatch_prob_2loop,
                            );
                            // Count 2-loop terminal mismatches.
                            logsumexp(
                              &mut alignfold_counts_expected.terminal_mismatch_scores[basepair4.0]
                                [basepair4.1][mismatch_pair4.0][mismatch_pair4.1],
                              pairmatch_prob_2loop,
                            );
                            logsumexp(
                              &mut alignfold_counts_expected.terminal_mismatch_scores[basepair2.1]
                                [basepair2.0][mismatch_pair2.1][mismatch_pair2.0],
                              pairmatch_prob_2loop,
                            );
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            let sum_ratio = alignfold_sums.sums_accessible_multibranch[&pos_quad] - sum_close;
            for (pos_pair, forward_sums) in &alignfold_sums.forward_sums_hashed_poss {
              let &(u, v) = pos_pair;
              if !(u < i && v < k) {
                continue;
              }
              let forward_sums2 = &alignfold_sums.forward_sums_hashed_poss2[pos_pair];
              let pos_quad2 = (u, j, v, l);
              let mut forward_term = NEG_INFINITY;
              let mut forward_term_match = forward_term;
              let mut forward_term2 = forward_term;
              let mut forward_term_match2 = forward_term;
              let pos_pair2 = (i - T::one(), k - T::one());
              if let Some(x) = forward_sums2.get(&pos_pair2) {
                logsumexp(&mut forward_term, x.sum_1ormore_pairmatches);
                logsumexp(&mut forward_term2, x.sum_seqalign_multibranch);
              }
              if trains_alignfold_scores {
                if let Some(x) = forward_sums.get(&pos_pair2) {
                  let y = x.sum_1ormore_pairmatches + alignfold_scores.match2match_score;
                  logsumexp(&mut forward_term_match, y);
                  let y = x.sum_seqalign_multibranch + alignfold_scores.match2match_score;
                  logsumexp(&mut forward_term_match2, y);
                }
              }
              let mut sum_multibranch = NEG_INFINITY;
              if let Some(x) = prob_coeffs_multibranch.get(&pos_quad2) {
                let x = x + sum_ratio;
                let y = x + forward_term;
                logsumexp(&mut sum_multibranch, y);
                if trains_alignfold_scores {
                  let y = prob_coeff + x + forward_term_match;
                  logsumexp(&mut alignfold_counts_expected.match2match_score, y);
                }
              }
              if let Some(x) = prob_coeffs_multibranch2.get(&pos_quad2) {
                let x = x + sum_ratio;
                let y = x + forward_term2;
                logsumexp(&mut sum_multibranch, y);
                if trains_alignfold_scores {
                  let y = prob_coeff + x + forward_term_match2;
                  logsumexp(&mut alignfold_counts_expected.match2match_score, y);
                }
              }
              if sum_multibranch > NEG_INFINITY {
                logsumexp(&mut sum, sum_multibranch);
                let pairmatch_prob_multibranch = prob_coeff + sum_multibranch;
                if trains_alignfold_scores {
                  // Count multi-loop terminal mismatches.
                  logsumexp(
                    &mut alignfold_counts_expected.dangling_scores_left[basepair.1][basepair.0]
                      [mismatch_pair.1],
                    pairmatch_prob_multibranch,
                  );
                  logsumexp(
                    &mut alignfold_counts_expected.dangling_scores_right[basepair.1][basepair.0]
                      [mismatch_pair.0],
                    pairmatch_prob_multibranch,
                  );
                  logsumexp(
                    &mut alignfold_counts_expected.dangling_scores_left[basepair2.1][basepair2.0]
                      [mismatch_pair2.1],
                    pairmatch_prob_multibranch,
                  );
                  logsumexp(
                    &mut alignfold_counts_expected.dangling_scores_right[basepair2.1][basepair2.0]
                      [mismatch_pair2.0],
                    pairmatch_prob_multibranch,
                  );
                  // Count helix ends.
                  logsumexp(
                    &mut alignfold_counts_expected.helix_close_scores[basepair.1][basepair.0],
                    pairmatch_prob_multibranch,
                  );
                  logsumexp(
                    &mut alignfold_counts_expected.helix_close_scores[basepair2.1][basepair2.0],
                    pairmatch_prob_multibranch,
                  );
                  // Count multi-loop closings.
                  logsumexp(
                    &mut alignfold_counts_expected.multibranch_score_base,
                    (2. as Prob).ln() + pairmatch_prob_multibranch,
                  );
                  // Count multi-loop closing basepairings.
                  logsumexp(
                    &mut alignfold_counts_expected.multibranch_score_basepair,
                    (2. as Prob).ln() + pairmatch_prob_multibranch,
                  );
                  // Count multi-loop accessible basepairings.
                  logsumexp(
                    &mut alignfold_counts_expected.multibranch_score_basepair,
                    (2. as Prob).ln() + pairmatch_prob_multibranch,
                  );
                }
              }
            }
            if sum > NEG_INFINITY {
              alignfold_outside_sums.insert(pos_quad, sum);
              let pairmatch_prob = prob_coeff + sum;
              if produces_match_probs {
                alignfold_probs
                  .pairmatch_probs
                  .insert(pos_quad, pairmatch_prob);
                match alignfold_probs.match_probs.get_mut(&(i, k)) {
                  Some(x) => {
                    logsumexp(x, pairmatch_prob);
                  }
                  None => {
                    alignfold_probs.match_probs.insert((i, k), pairmatch_prob);
                  }
                }
                match alignfold_probs.match_probs.get_mut(&(j, l)) {
                  Some(x) => {
                    logsumexp(x, pairmatch_prob);
                  }
                  None => {
                    alignfold_probs.match_probs.insert((j, l), pairmatch_prob);
                  }
                }
              }
              if trains_alignfold_scores {
                // Count base pairs.
                let dict_min_basepair = get_dict_min_pair(&basepair);
                let dict_min_basepair2 = get_dict_min_pair(&basepair2);
                logsumexp(
                  &mut alignfold_counts_expected.basepair_scores[dict_min_basepair.0]
                    [dict_min_basepair.1],
                  pairmatch_prob,
                );
                logsumexp(
                  &mut alignfold_counts_expected.basepair_scores[dict_min_basepair2.0]
                    [dict_min_basepair2.1],
                  pairmatch_prob,
                );
                // Count alignments.
                let dict_min_match = get_dict_min_pair(&(basepair.0, basepair2.0));
                logsumexp(
                  &mut alignfold_counts_expected.match_scores[dict_min_match.0][dict_min_match.1],
                  pairmatch_prob,
                );
                let dict_min_match = get_dict_min_pair(&(basepair.1, basepair2.1));
                logsumexp(
                  &mut alignfold_counts_expected.match_scores[dict_min_match.0][dict_min_match.1],
                  pairmatch_prob,
                );
              }
              match alignfold_probs.basepair_probs_pair.0.get_mut(&(i, j)) {
                Some(x) => {
                  logsumexp(x, pairmatch_prob);
                }
                None => {
                  alignfold_probs
                    .basepair_probs_pair
                    .0
                    .insert((i, j), pairmatch_prob);
                }
              }
              match alignfold_probs.basepair_probs_pair.1.get_mut(&(k, l)) {
                Some(x) => {
                  logsumexp(x, pairmatch_prob);
                }
                None => {
                  alignfold_probs
                    .basepair_probs_pair
                    .1
                    .insert((k, l), pairmatch_prob);
                }
              }
              if produces_struct_profs {
                logsumexp(
                  &mut alignfold_probs.context_profs_pair.0[(long_i, CONTEXT_INDEX_BASEPAIR)],
                  pairmatch_prob,
                );
                logsumexp(
                  &mut alignfold_probs.context_profs_pair.0[(long_j, CONTEXT_INDEX_BASEPAIR)],
                  pairmatch_prob,
                );
                logsumexp(
                  &mut alignfold_probs.context_profs_pair.1[(long_k, CONTEXT_INDEX_BASEPAIR)],
                  pairmatch_prob,
                );
                logsumexp(
                  &mut alignfold_probs.context_profs_pair.1[(long_l, CONTEXT_INDEX_BASEPAIR)],
                  pairmatch_prob,
                );
              }
              let pairmatch_score = alignfold_scores.match_scores[basepair.0][basepair2.0]
                + alignfold_scores.match_scores[basepair.1][basepair2.1];
              let multibranch_close_score = fold_scores_pair.0.multibranch_close_scores[&(i, j)];
              let multibranch_close_score2 = fold_scores_pair.1.multibranch_close_scores[&(k, l)];
              if trains_alignfold_scores {
                let mismatch_pair = (seq_pair.0[long_i + 1], seq_pair.0[long_j - 1]);
                let mismatch_pair2 = (seq_pair.1[long_k + 1], seq_pair.1[long_l - 1]);
                let forward_sums = &alignfold_sums.forward_sums_hashed_poss[&(i, k)];
                let forward_sums2 = &alignfold_sums.forward_sums_hashed_poss2[&(i, k)];
                if substr_len.to_usize().unwrap() - 2 <= MAX_LOOP_LEN
                  && substr_len2.to_usize().unwrap() - 2 <= MAX_LOOP_LEN
                {
                  let mut sum_seqalign = NEG_INFINITY;
                  let mut sum_seqalign_match = sum_seqalign;
                  let pos_pair2 = (j - T::one(), l - T::one());
                  if let Some(x) = forward_sums2.get(&pos_pair2) {
                    logsumexp(&mut sum_seqalign, x.sum_seqalign);
                  }
                  if let Some(x) = forward_sums.get(&pos_pair2) {
                    let x = x.sum_seqalign + alignfold_scores.match2match_score;
                    logsumexp(&mut sum_seqalign_match, x);
                  }
                  let hairpin_score = fold_scores_pair.0.hairpin_scores[&(i, j)];
                  let hairpin_score2 = fold_scores_pair.1.hairpin_scores[&(k, l)];
                  let prob = sum - global_sum
                    + sum_seqalign_match
                    + hairpin_score
                    + hairpin_score2
                    + pairmatch_score;
                  logsumexp(&mut alignfold_counts_expected.match2match_score, prob);
                  let pairmatch_prob_hairpin = sum - global_sum
                    + sum_seqalign
                    + hairpin_score
                    + hairpin_score2
                    + pairmatch_score;
                  logsumexp(
                    &mut alignfold_counts_expected.hairpin_scores_len[long_j - long_i - 1],
                    pairmatch_prob_hairpin,
                  );
                  logsumexp(
                    &mut alignfold_counts_expected.hairpin_scores_len[long_l - long_k - 1],
                    pairmatch_prob_hairpin,
                  );
                  logsumexp(
                    &mut alignfold_counts_expected.terminal_mismatch_scores[basepair.0][basepair.1]
                      [mismatch_pair.0][mismatch_pair.1],
                    pairmatch_prob_hairpin,
                  );
                  logsumexp(
                    &mut alignfold_counts_expected.terminal_mismatch_scores[basepair2.0]
                      [basepair2.1][mismatch_pair2.0][mismatch_pair2.1],
                    pairmatch_prob_hairpin,
                  );
                  // Count helix ends.
                  logsumexp(
                    &mut alignfold_counts_expected.helix_close_scores[basepair.0][basepair.1],
                    pairmatch_prob_hairpin,
                  );
                  logsumexp(
                    &mut alignfold_counts_expected.helix_close_scores[basepair2.0][basepair2.1],
                    pairmatch_prob_hairpin,
                  );
                }
                let mut sum_multibranch = NEG_INFINITY;
                let mut sum_multibranch_match = sum_multibranch;
                if let Some(x) = forward_sums2.get(&pos_pair2) {
                  logsumexp(&mut sum_multibranch, x.sum_multibranch);
                }
                if let Some(x) = forward_sums.get(&pos_pair2) {
                  let x = x.sum_multibranch + alignfold_scores.match2match_score;
                  logsumexp(&mut sum_multibranch_match, x);
                }
                let prob = sum - global_sum
                  + sum_multibranch_match
                  + multibranch_close_score
                  + multibranch_close_score2
                  + pairmatch_score;
                logsumexp(&mut alignfold_counts_expected.match2match_score, prob);
                let pairmatch_prob_multibranch = sum - global_sum
                  + sum_multibranch
                  + multibranch_close_score
                  + multibranch_close_score2
                  + pairmatch_score;
                // Count multi-loop terminal mismatches.
                logsumexp(
                  &mut alignfold_counts_expected.dangling_scores_left[basepair.0][basepair.1]
                    [mismatch_pair.0],
                  pairmatch_prob_multibranch,
                );
                logsumexp(
                  &mut alignfold_counts_expected.dangling_scores_right[basepair.0][basepair.1]
                    [mismatch_pair.1],
                  pairmatch_prob_multibranch,
                );
                logsumexp(
                  &mut alignfold_counts_expected.dangling_scores_left[basepair2.0][basepair2.1]
                    [mismatch_pair2.0],
                  pairmatch_prob_multibranch,
                );
                logsumexp(
                  &mut alignfold_counts_expected.dangling_scores_right[basepair2.0][basepair2.1]
                    [mismatch_pair2.1],
                  pairmatch_prob_multibranch,
                );
                // Count helix ends.
                logsumexp(
                  &mut alignfold_counts_expected.helix_close_scores[basepair.0][basepair.1],
                  pairmatch_prob_multibranch,
                );
                logsumexp(
                  &mut alignfold_counts_expected.helix_close_scores[basepair2.0][basepair2.1],
                  pairmatch_prob_multibranch,
                );
              }
              let coeff =
                sum + pairmatch_score + multibranch_close_score + multibranch_close_score2;
              let backward_sums = &alignfold_sums.backward_sums_hashed_poss2[&(j, l)];
              for pos_pair in match_probs.keys() {
                let &(u, v) = pos_pair;
                if !(i < u && u < j && k < v && v < l) {
                  continue;
                }
                let mut backward_term = NEG_INFINITY;
                let mut backward_term2 = backward_term;
                let pos_pair2 = (u + T::one(), v + T::one());
                if let Some(x) = backward_sums.get(&pos_pair2) {
                  logsumexp(&mut backward_term, x.sum_0ormore_pairmatches);
                  logsumexp(&mut backward_term2, x.sum_1ormore_pairmatches);
                }
                let pos_quad2 = (i, u, k, v);
                let x = coeff + backward_term;
                match prob_coeffs_multibranch.get_mut(&pos_quad2) {
                  Some(y) => {
                    logsumexp(y, x);
                  }
                  None => {
                    prob_coeffs_multibranch.insert(pos_quad2, x);
                  }
                }
                let x = coeff + backward_term2;
                match prob_coeffs_multibranch2.get_mut(&pos_quad2) {
                  Some(y) => {
                    logsumexp(y, x);
                  }
                  None => {
                    prob_coeffs_multibranch2.insert(pos_quad2, x);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  for x in alignfold_probs.basepair_probs_pair.0.values_mut() {
    *x = expf(*x);
  }
  for x in alignfold_probs.basepair_probs_pair.1.values_mut() {
    *x = expf(*x);
  }
  let needs_twoloop_sums = produces_match_probs || trains_alignfold_scores;
  let needs_indel_info = produces_struct_profs || trains_alignfold_scores;
  if produces_struct_profs || needs_twoloop_sums {
    let mut unpair_probs_range_external =
      (SparseProbMat::<T>::default(), SparseProbMat::<T>::default());
    let mut unpair_probs_range_hairpin =
      (SparseProbMat::<T>::default(), SparseProbMat::<T>::default());
    let mut unpair_probs_range = (SparseProbMat::<T>::default(), SparseProbMat::<T>::default());
    for u in range(T::zero(), seq_len_pair.0 - T::one()) {
      let long_u = u.to_usize().unwrap();
      let base = seq_pair.0[long_u];
      for v in range(T::zero(), seq_len_pair.1 - T::one()) {
        let pos_pair = (u, v);
        let long_v = v.to_usize().unwrap();
        let base2 = seq_pair.1[long_v];
        let pos_pair2 = (u + T::one(), v + T::one());
        let long_pos_pair2 = (
          pos_pair2.0.to_usize().unwrap(),
          pos_pair2.1.to_usize().unwrap(),
        );
        let dict_min_match = get_dict_min_pair(&(base, base2));
        if match_probs.contains_key(&pos_pair) {
          let pos_pair_loopmatch = (u - T::one(), v - T::one());
          let mut loopmatch_prob_external = NEG_INFINITY;
          let loopmatch_score = alignfold_scores.match_scores[base][base2]
            + 2. * alignfold_scores.external_score_unpair;
          let mut backward_term = NEG_INFINITY;
          if let Some(&x) = alignfold_sums.backward_sums_external2.get(&pos_pair2) {
            logsumexp(&mut backward_term, x);
          }
          if let Some(&x) = alignfold_sums
            .forward_sums_external2
            .get(&pos_pair_loopmatch)
          {
            let term = loopmatch_score + x + backward_term - global_sum;
            logsumexp(&mut loopmatch_prob_external, term);
          }
          if produces_struct_profs {
            logsumexp(
              &mut alignfold_probs.context_profs_pair.0[(long_u, CONTEXT_INDEX_EXTERNAL)],
              loopmatch_prob_external,
            );
            logsumexp(
              &mut alignfold_probs.context_profs_pair.1[(long_v, CONTEXT_INDEX_EXTERNAL)],
              loopmatch_prob_external,
            );
          }
          if produces_match_probs {
            match alignfold_probs.loopmatch_probs.get_mut(&pos_pair) {
              Some(x) => {
                logsumexp(x, loopmatch_prob_external);
              }
              None => {
                alignfold_probs
                  .loopmatch_probs
                  .insert(pos_pair, loopmatch_prob_external);
              }
            }
          }
          if trains_alignfold_scores {
            logsumexp(
              &mut alignfold_counts_expected.match_scores[dict_min_match.0][dict_min_match.1],
              loopmatch_prob_external,
            );
            logsumexp(
              &mut alignfold_counts_expected.external_score_unpair,
              (2. as Prob).ln() + loopmatch_prob_external,
            );
            if let Some(&x) = alignfold_sums
              .forward_sums_external
              .get(&pos_pair_loopmatch)
            {
              let begins_sum = pos_pair_loopmatch == leftmost_pos_pair;
              let x = x
                + if begins_sum {
                  alignfold_scores.init_match_score
                } else {
                  alignfold_scores.match2match_score
                };
              let x = loopmatch_score + x + backward_term - global_sum;
              if begins_sum {
                logsumexp(&mut alignfold_counts_expected.init_match_score, x);
              } else {
                logsumexp(&mut alignfold_counts_expected.match2match_score, x);
              }
            }
          }
        }
        if needs_indel_info {
          if let Some(&sum) = alignfold_sums.forward_sums_external.get(&pos_pair) {
            let begins_sum = pos_pair == leftmost_pos_pair;
            let forward_term = sum
              + if begins_sum {
                alignfold_scores.init_insert_score
              } else {
                alignfold_scores.match2insert_score
              };
            if let Some(x) = matchable_poss.get(&pos_pair2.0) {
              for &x in x {
                if x <= pos_pair2.1 {
                  continue;
                }
                let pos_pair3 = (pos_pair2.0, x);
                if let Some(&y) = alignfold_sums.backward_sums_external.get(&pos_pair3) {
                  let long_x = x.to_usize().unwrap();
                  let ends_sum = pos_pair3 == rightmost_pos_pair;
                  let z = range_insert_scores.insert_scores_external2[long_pos_pair2.1][long_x - 1]
                    + if ends_sum {
                      0.
                    } else {
                      alignfold_scores.match2insert_score
                    };
                  let z = forward_term + y + z - global_sum;
                  if trains_alignfold_scores {
                    if begins_sum {
                      logsumexp(&mut alignfold_counts_expected.init_insert_score, z);
                    } else {
                      logsumexp(&mut alignfold_counts_expected.match2insert_score, z);
                    }
                    if !ends_sum {
                      logsumexp(&mut alignfold_counts_expected.match2insert_score, z);
                    }
                    logsumexp(
                      &mut alignfold_counts_expected.external_score_unpair,
                      ((long_x - long_pos_pair2.1) as Prob).ln() + z,
                    );
                    if pos_pair2.1 < x - T::one() {
                      logsumexp(
                        &mut alignfold_counts_expected.insert_extend_score,
                        ((long_x - long_pos_pair2.1 - 1) as Prob).ln() + z,
                      );
                    }
                  }
                  let pos_pair4 = (pos_pair2.1, x - T::one());
                  if produces_struct_profs {
                    match unpair_probs_range_external.1.get_mut(&pos_pair4) {
                      Some(x) => {
                        logsumexp(x, z);
                      }
                      None => {
                        unpair_probs_range_external.1.insert(pos_pair4, z);
                      }
                    }
                  }
                  if trains_alignfold_scores {
                    match unpair_probs_range.1.get_mut(&pos_pair4) {
                      Some(x) => {
                        logsumexp(x, z);
                      }
                      None => {
                        unpair_probs_range.1.insert(pos_pair4, z);
                      }
                    }
                  }
                }
              }
            }
            if let Some(x) = matchable_poss2.get(&pos_pair2.1) {
              for &x in x {
                if x <= pos_pair2.0 {
                  continue;
                }
                let pos_pair3 = (x, pos_pair2.1);
                if let Some(&y) = alignfold_sums.backward_sums_external.get(&pos_pair3) {
                  let long_x = x.to_usize().unwrap();
                  let ends_sum = pos_pair3 == rightmost_pos_pair;
                  let z = range_insert_scores.insert_scores_external[long_pos_pair2.0][long_x - 1]
                    + if ends_sum {
                      0.
                    } else {
                      alignfold_scores.match2insert_score
                    };
                  let z = forward_term + y + z - global_sum;
                  if trains_alignfold_scores {
                    if begins_sum {
                      logsumexp(&mut alignfold_counts_expected.init_insert_score, z);
                    } else {
                      logsumexp(&mut alignfold_counts_expected.match2insert_score, z);
                    }
                    if !ends_sum {
                      logsumexp(&mut alignfold_counts_expected.match2insert_score, z);
                    }
                    logsumexp(
                      &mut alignfold_counts_expected.external_score_unpair,
                      ((long_x - long_pos_pair2.0) as Prob).ln() + z,
                    );
                    if pos_pair2.0 < x - T::one() {
                      logsumexp(
                        &mut alignfold_counts_expected.insert_extend_score,
                        ((long_x - long_pos_pair2.0 - 1) as Prob).ln() + z,
                      );
                    }
                  }
                  let pos_pair4 = (pos_pair2.0, x - T::one());
                  if produces_struct_profs {
                    match unpair_probs_range_external.0.get_mut(&pos_pair4) {
                      Some(x) => {
                        logsumexp(x, z);
                      }
                      None => {
                        unpair_probs_range_external.0.insert(pos_pair4, z);
                      }
                    }
                  }
                  if trains_alignfold_scores {
                    match unpair_probs_range.0.get_mut(&pos_pair4) {
                      Some(x) => {
                        logsumexp(x, z);
                      }
                      None => {
                        unpair_probs_range.0.insert(pos_pair4, z);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    for (pos_quad, &outside_sum) in &alignfold_outside_sums {
      let (i, j, k, l) = *pos_quad;
      let (long_i, long_j, long_k, long_l) = (
        i.to_usize().unwrap(),
        j.to_usize().unwrap(),
        k.to_usize().unwrap(),
        l.to_usize().unwrap(),
      );
      let basepair = (seq_pair.0[long_i], seq_pair.0[long_j]);
      let basepair2 = (seq_pair.1[long_k], seq_pair.1[long_l]);
      let hairpin_score = if j - i - T::one() <= T::from_usize(MAX_LOOP_LEN).unwrap() {
        fold_scores_pair.0.hairpin_scores[&(i, j)]
      } else {
        NEG_INFINITY
      };
      let hairpin_score2 = if l - k - T::one() <= T::from_usize(MAX_LOOP_LEN).unwrap() {
        fold_scores_pair.1.hairpin_scores[&(k, l)]
      } else {
        NEG_INFINITY
      };
      let multibranch_close_score = fold_scores_pair.0.multibranch_close_scores[&(i, j)];
      let multibranch_close_score2 = fold_scores_pair.1.multibranch_close_scores[&(k, l)];
      let pairmatch_score = alignfold_scores.match_scores[basepair.0][basepair2.0]
        + alignfold_scores.match_scores[basepair.1][basepair2.1];
      let prob_coeff = outside_sum - global_sum + pairmatch_score;
      let forward_sums = &alignfold_sums.forward_sums_hashed_poss[&(i, k)];
      let forward_sums2 = &alignfold_sums.forward_sums_hashed_poss2[&(i, k)];
      let backward_sums = &alignfold_sums.backward_sums_hashed_poss[&(j, l)];
      let backward_sums2 = &alignfold_sums.backward_sums_hashed_poss2[&(j, l)];
      let (forward_sums_2loop, forward_sums_2loop2) = if needs_twoloop_sums {
        let computes_forward_sums = true;
        get_2loop_sums((
          seq_pair,
          alignfold_scores,
          match_probs,
          pos_quad,
          alignfold_sums,
          computes_forward_sums,
          forward_pos_pairs,
          fold_scores_pair,
          range_insert_scores,
          matchable_poss,
          matchable_poss2,
        ))
      } else {
        (SparseSumMat::<T>::default(), SparseSumMat::<T>::default())
      };
      let (backward_sums_2loop, backward_sums_2loop2) = if needs_twoloop_sums {
        let computes_forward_sums = false;
        get_2loop_sums((
          seq_pair,
          alignfold_scores,
          match_probs,
          pos_quad,
          alignfold_sums,
          computes_forward_sums,
          backward_pos_pairs,
          fold_scores_pair,
          range_insert_scores,
          matchable_poss,
          matchable_poss2,
        ))
      } else {
        (SparseSumMat::<T>::default(), SparseSumMat::<T>::default())
      };
      if trains_alignfold_scores {
        let pos_pair2 = (j - T::one(), l - T::one());
        if let Some(&x) = forward_sums_2loop.get(&pos_pair2) {
          let x = prob_coeff + x + alignfold_scores.match2match_score;
          logsumexp(&mut alignfold_counts_expected.match2match_score, x);
        }
      }
      for u in range_inclusive(i, j) {
        let long_u = u.to_usize().unwrap();
        let base = seq_pair.0[long_u];
        for v in range_inclusive(k, l) {
          let pos_pair = (u, v);
          let long_v = v.to_usize().unwrap();
          let base2 = seq_pair.1[long_v];
          let pos_pair2 = (u + T::one(), v + T::one());
          let long_pos_pair2 = (
            pos_pair2.0.to_usize().unwrap(),
            pos_pair2.1.to_usize().unwrap(),
          );
          let dict_min_match = get_dict_min_pair(&(base, base2));
          let mut backward_term_seqalign = NEG_INFINITY;
          let mut backward_term_multibranch = backward_term_seqalign;
          let mut backward_term_1ormore_pairmatches = backward_term_seqalign;
          let mut backward_term_0ormore_pairmatches = backward_term_seqalign;
          let mut backward_term_2loop = backward_term_seqalign;
          if let Some(x) = backward_sums2.get(&pos_pair2) {
            logsumexp(&mut backward_term_seqalign, x.sum_seqalign);
            if needs_twoloop_sums {
              logsumexp(&mut backward_term_multibranch, x.sum_multibranch);
              logsumexp(
                &mut backward_term_1ormore_pairmatches,
                x.sum_1ormore_pairmatches,
              );
              logsumexp(
                &mut backward_term_0ormore_pairmatches,
                x.sum_0ormore_pairmatches,
              );
            }
          }
          if needs_twoloop_sums {
            if let Some(&x) = backward_sums_2loop2.get(&pos_pair2) {
              logsumexp(&mut backward_term_2loop, x);
            }
          }
          let prob_coeff_hairpin = prob_coeff + hairpin_score + hairpin_score2;
          let prob_coeff_multibranch =
            prob_coeff + multibranch_close_score + multibranch_close_score2;
          if match_probs.contains_key(&pos_pair) {
            let pos_pair_loopmatch = (u - T::one(), v - T::one());
            let loopmatch_score = alignfold_scores.match_scores[base][base2];
            let loopmatch_score_multibranch =
              loopmatch_score + 2. * alignfold_scores.multibranch_score_unpair;
            let mut loopmatch_prob_hairpin = NEG_INFINITY;
            let mut loopmatch_prob_multibranch = loopmatch_prob_hairpin;
            let mut loopmatch_prob_2loop = loopmatch_prob_hairpin;
            if let Some(x) = forward_sums2.get(&pos_pair_loopmatch) {
              let y =
                prob_coeff_hairpin + loopmatch_score + x.sum_seqalign + backward_term_seqalign;
              logsumexp(&mut loopmatch_prob_hairpin, y);
              if needs_twoloop_sums {
                let y = prob_coeff_multibranch
                  + loopmatch_score_multibranch
                  + x.sum_seqalign_multibranch
                  + backward_term_multibranch;
                logsumexp(&mut loopmatch_prob_multibranch, y);
                let y = prob_coeff_multibranch
                  + loopmatch_score_multibranch
                  + x.sum_1st_pairmatches
                  + backward_term_1ormore_pairmatches;
                logsumexp(&mut loopmatch_prob_multibranch, y);
                let y = prob_coeff_multibranch
                  + loopmatch_score_multibranch
                  + x.sum_multibranch
                  + backward_term_0ormore_pairmatches;
                logsumexp(&mut loopmatch_prob_multibranch, y);
                let y = prob_coeff + loopmatch_score + x.sum_seqalign + backward_term_2loop;
                logsumexp(&mut loopmatch_prob_2loop, y);
              }
            }
            if produces_struct_profs {
              logsumexp(
                &mut alignfold_probs.context_profs_pair.0[(long_u, CONTEXT_INDEX_HAIRPIN)],
                loopmatch_prob_hairpin,
              );
              logsumexp(
                &mut alignfold_probs.context_profs_pair.1[(long_v, CONTEXT_INDEX_HAIRPIN)],
                loopmatch_prob_hairpin,
              );
            }
            if needs_twoloop_sums {
              if let Some(&x) = forward_sums_2loop2.get(&pos_pair_loopmatch) {
                let x = prob_coeff + loopmatch_score + x + backward_term_seqalign;
                logsumexp(&mut loopmatch_prob_2loop, x);
              }
              let mut prob = NEG_INFINITY;
              logsumexp(&mut prob, loopmatch_prob_hairpin);
              logsumexp(&mut prob, loopmatch_prob_multibranch);
              logsumexp(&mut prob, loopmatch_prob_2loop);
              if produces_match_probs {
                match alignfold_probs.loopmatch_probs.get_mut(&pos_pair) {
                  Some(x) => {
                    logsumexp(x, prob);
                  }
                  None => {
                    alignfold_probs.loopmatch_probs.insert(pos_pair, prob);
                  }
                }
              }
              if trains_alignfold_scores {
                logsumexp(
                  &mut alignfold_counts_expected.match_scores[dict_min_match.0][dict_min_match.1],
                  prob,
                );
                logsumexp(
                  &mut alignfold_counts_expected.multibranch_score_unpair,
                  (2. as Prob).ln() + loopmatch_prob_multibranch,
                );
              }
            }
            if let Some(x) = forward_sums.get(&pos_pair_loopmatch) {
              let y = prob_coeff_hairpin
                + loopmatch_score
                + x.sum_seqalign
                + alignfold_scores.match2match_score
                + backward_term_seqalign;
              if trains_alignfold_scores {
                logsumexp(&mut alignfold_counts_expected.match2match_score, y);
              }
              if needs_twoloop_sums {
                let y = prob_coeff_multibranch
                  + loopmatch_score_multibranch
                  + x.sum_seqalign_multibranch
                  + alignfold_scores.match2match_score
                  + backward_term_multibranch;
                if trains_alignfold_scores {
                  logsumexp(&mut alignfold_counts_expected.match2match_score, y);
                }
                let y = prob_coeff_multibranch
                  + loopmatch_score_multibranch
                  + x.sum_1st_pairmatches
                  + alignfold_scores.match2match_score
                  + backward_term_1ormore_pairmatches;
                if trains_alignfold_scores {
                  logsumexp(&mut alignfold_counts_expected.match2match_score, y);
                }
                let y = prob_coeff_multibranch
                  + loopmatch_score_multibranch
                  + x.sum_multibranch
                  + alignfold_scores.match2match_score
                  + backward_term_0ormore_pairmatches;
                if trains_alignfold_scores {
                  logsumexp(&mut alignfold_counts_expected.match2match_score, y);
                }
                let y = prob_coeff
                  + loopmatch_score
                  + x.sum_seqalign
                  + alignfold_scores.match2match_score
                  + backward_term_2loop;
                if trains_alignfold_scores {
                  logsumexp(&mut alignfold_counts_expected.match2match_score, y);
                }
              }
            }
            if needs_indel_info {
              if let Some(sums) = forward_sums.get(&pos_pair) {
                let forward_term_seqalign = sums.sum_seqalign + alignfold_scores.match2insert_score;
                let forward_term_seqalign_multibranch =
                  sums.sum_seqalign_multibranch + alignfold_scores.match2insert_score;
                let forward_term_1st_pairmatches =
                  sums.sum_1st_pairmatches + alignfold_scores.match2insert_score;
                let forward_term_multibranch =
                  sums.sum_multibranch + alignfold_scores.match2insert_score;
                if let Some(x) = matchable_poss.get(&pos_pair2.0) {
                  for &x in x {
                    if x <= pos_pair2.1 {
                      continue;
                    }
                    let mut insert_prob = NEG_INFINITY;
                    let mut insert_prob_multibranch = insert_prob;
                    let long_x = x.to_usize().unwrap();
                    let y = range_insert_scores.insert_scores2[long_pos_pair2.1][long_x - 1]
                      + alignfold_scores.match2insert_score;
                    if let Some(z) = backward_sums.get(&(pos_pair2.0, x)) {
                      let a = prob_coeff_hairpin + forward_term_seqalign + y + z.sum_seqalign;
                      logsumexp(&mut insert_prob, a);
                      if produces_struct_profs {
                        let pos_pair4 = (pos_pair2.1, x - T::one());
                        match unpair_probs_range_hairpin.1.get_mut(&pos_pair4) {
                          Some(x) => {
                            logsumexp(x, a);
                          }
                          None => {
                            unpair_probs_range_hairpin.1.insert(pos_pair4, a);
                          }
                        }
                      }
                      if trains_alignfold_scores {
                        let y = range_insert_scores.insert_scores_multibranch2[long_pos_pair2.1]
                          [long_x - 1]
                          + alignfold_scores.match2insert_score;
                        let a = prob_coeff_multibranch
                          + forward_term_seqalign_multibranch
                          + y
                          + z.sum_multibranch;
                        logsumexp(&mut insert_prob_multibranch, a);
                        let a = prob_coeff_multibranch
                          + forward_term_1st_pairmatches
                          + y
                          + z.sum_1ormore_pairmatches;
                        logsumexp(&mut insert_prob_multibranch, a);
                        let a = prob_coeff_multibranch
                          + forward_term_multibranch
                          + y
                          + z.sum_0ormore_pairmatches;
                        logsumexp(&mut insert_prob_multibranch, a);
                        logsumexp(&mut insert_prob, insert_prob_multibranch);
                      }
                    }
                    if trains_alignfold_scores {
                      if let Some(&z) = backward_sums_2loop.get(&(pos_pair2.0, x)) {
                        let z = prob_coeff + forward_term_seqalign + y + z;
                        logsumexp(&mut insert_prob, z);
                      }
                      logsumexp(
                        &mut alignfold_counts_expected.match2insert_score,
                        (2. as Prob).ln() + insert_prob,
                      );
                      let pos_pair4 = (pos_pair2.1, x - T::one());
                      match unpair_probs_range.1.get_mut(&pos_pair4) {
                        Some(x) => {
                          logsumexp(x, insert_prob);
                        }
                        None => {
                          unpair_probs_range.1.insert(pos_pair4, insert_prob);
                        }
                      }
                      logsumexp(
                        &mut alignfold_counts_expected.multibranch_score_unpair,
                        ((long_x - long_pos_pair2.1) as Prob).ln() + insert_prob_multibranch,
                      );
                      if pos_pair2.1 < x - T::one() {
                        logsumexp(
                          &mut alignfold_counts_expected.insert_extend_score,
                          ((long_x - long_pos_pair2.1 - 1) as Prob).ln() + insert_prob,
                        );
                      }
                    }
                  }
                }
                if let Some(x) = matchable_poss2.get(&pos_pair2.1) {
                  for &x in x {
                    if x <= pos_pair2.0 {
                      continue;
                    }
                    let mut insert_prob = NEG_INFINITY;
                    let mut insert_prob_multibranch = insert_prob;
                    let long_x = x.to_usize().unwrap();
                    let y = range_insert_scores.insert_scores[long_pos_pair2.0][long_x - 1]
                      + alignfold_scores.match2insert_score;
                    if let Some(z) = backward_sums.get(&(x, pos_pair2.1)) {
                      let a = prob_coeff_hairpin + forward_term_seqalign + y + z.sum_seqalign;
                      logsumexp(&mut insert_prob, a);
                      if produces_struct_profs {
                        let pos_pair4 = (pos_pair2.0, x - T::one());
                        match unpair_probs_range_hairpin.0.get_mut(&pos_pair4) {
                          Some(x) => {
                            logsumexp(x, a);
                          }
                          None => {
                            unpair_probs_range_hairpin.0.insert(pos_pair4, a);
                          }
                        }
                      }
                      if trains_alignfold_scores {
                        let y = range_insert_scores.insert_scores_multibranch[long_pos_pair2.0]
                          [long_x - 1]
                          + alignfold_scores.match2insert_score;
                        let a = prob_coeff_multibranch
                          + forward_term_seqalign_multibranch
                          + y
                          + z.sum_multibranch;
                        logsumexp(&mut insert_prob_multibranch, a);
                        let a = prob_coeff_multibranch
                          + forward_term_1st_pairmatches
                          + y
                          + z.sum_1ormore_pairmatches;
                        logsumexp(&mut insert_prob_multibranch, a);
                        let a = prob_coeff_multibranch
                          + forward_term_multibranch
                          + y
                          + z.sum_0ormore_pairmatches;
                        logsumexp(&mut insert_prob_multibranch, a);
                        logsumexp(&mut insert_prob, insert_prob_multibranch);
                      }
                    }
                    if let Some(&z) = backward_sums_2loop.get(&(x, pos_pair2.1)) {
                      let z = prob_coeff + forward_term_seqalign + y + z;
                      logsumexp(&mut insert_prob, z);
                    }
                    if trains_alignfold_scores {
                      logsumexp(
                        &mut alignfold_counts_expected.match2insert_score,
                        (2. as Prob).ln() + insert_prob,
                      );
                      let pos_pair4 = (pos_pair2.0, x - T::one());
                      match unpair_probs_range.0.get_mut(&pos_pair4) {
                        Some(x) => {
                          logsumexp(x, insert_prob);
                        }
                        None => {
                          unpair_probs_range.0.insert(pos_pair4, insert_prob);
                        }
                      }
                      logsumexp(
                        &mut alignfold_counts_expected.multibranch_score_unpair,
                        ((long_x - long_pos_pair2.0) as Prob).ln() + insert_prob_multibranch,
                      );
                      if pos_pair2.0 < x - T::one() {
                        logsumexp(
                          &mut alignfold_counts_expected.insert_extend_score,
                          ((long_x - long_pos_pair2.0 - 1) as Prob).ln() + insert_prob,
                        );
                      }
                    }
                  }
                }
              }
              if trains_alignfold_scores {
                if let Some(&forward_sum) = forward_sums_2loop.get(&pos_pair) {
                  if let Some(x) = matchable_poss.get(&pos_pair2.0) {
                    for &x in x {
                      if x <= pos_pair2.1 {
                        continue;
                      }
                      let mut insert_prob = NEG_INFINITY;
                      let long_x = x.to_usize().unwrap();
                      let y = range_insert_scores.insert_scores2[long_pos_pair2.1][long_x - 1]
                        + alignfold_scores.match2insert_score;
                      if let Some(z) = backward_sums.get(&(pos_pair2.0, x)) {
                        let a = prob_coeff + forward_sum + y + z.sum_seqalign;
                        logsumexp(&mut insert_prob, a);
                        logsumexp(
                          &mut alignfold_counts_expected.match2insert_score,
                          (2. as Prob).ln() + insert_prob,
                        );
                        let pos_pair4 = (pos_pair2.1, x - T::one());
                        match unpair_probs_range.1.get_mut(&pos_pair4) {
                          Some(x) => {
                            logsumexp(x, insert_prob);
                          }
                          None => {
                            unpair_probs_range.1.insert(pos_pair4, insert_prob);
                          }
                        }
                        if pos_pair2.1 < x - T::one() {
                          logsumexp(
                            &mut alignfold_counts_expected.insert_extend_score,
                            ((long_x - long_pos_pair2.1 - 1) as Prob).ln() + insert_prob,
                          );
                        }
                      }
                    }
                  }
                  if let Some(x) = matchable_poss2.get(&pos_pair2.1) {
                    for &x in x {
                      if x <= pos_pair2.0 {
                        continue;
                      }
                      let mut insert_prob = NEG_INFINITY;
                      let long_x = x.to_usize().unwrap();
                      let y = range_insert_scores.insert_scores[long_pos_pair2.0][long_x - 1]
                        + alignfold_scores.match2insert_score;
                      if let Some(z) = backward_sums.get(&(x, pos_pair2.1)) {
                        let a = prob_coeff + forward_sum + y + z.sum_seqalign;
                        logsumexp(&mut insert_prob, a);
                        logsumexp(
                          &mut alignfold_counts_expected.match2insert_score,
                          (2. as Prob).ln() + insert_prob,
                        );
                        let pos_pair4 = (pos_pair2.0, x - T::one());
                        match unpair_probs_range.0.get_mut(&pos_pair4) {
                          Some(x) => {
                            logsumexp(x, insert_prob);
                          }
                          None => {
                            unpair_probs_range.0.insert(pos_pair4, insert_prob);
                          }
                        }
                        if pos_pair2.0 < x - T::one() {
                          logsumexp(
                            &mut alignfold_counts_expected.insert_extend_score,
                            ((long_x - long_pos_pair2.0 - 1) as Prob).ln() + insert_prob,
                          );
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    if needs_indel_info {
      for (x, &y) in &unpair_probs_range.0 {
        for i in range_inclusive(x.0, x.1) {
          let long_i = i.to_usize().unwrap();
          let x = seq_pair.0[long_i];
          logsumexp(&mut alignfold_counts_expected.insert_scores[x], y);
        }
      }
      for (x, &y) in &unpair_probs_range.1 {
        for i in range_inclusive(x.0, x.1) {
          let long_i = i.to_usize().unwrap();
          let x = seq_pair.1[long_i];
          logsumexp(&mut alignfold_counts_expected.insert_scores[x], y);
        }
      }
    }
    if produces_struct_profs {
      for (x, &y) in &unpair_probs_range_external.0 {
        for i in range_inclusive(x.0, x.1) {
          let long_i = i.to_usize().unwrap();
          logsumexp(
            &mut alignfold_probs.context_profs_pair.0[(long_i, CONTEXT_INDEX_EXTERNAL)],
            y,
          );
        }
      }
      for (x, &y) in &unpair_probs_range_external.1 {
        for i in range_inclusive(x.0, x.1) {
          let long_i = i.to_usize().unwrap();
          logsumexp(
            &mut alignfold_probs.context_profs_pair.1[(long_i, CONTEXT_INDEX_EXTERNAL)],
            y,
          );
        }
      }
      for (x, &y) in &unpair_probs_range_hairpin.0 {
        for i in range_inclusive(x.0, x.1) {
          let long_i = i.to_usize().unwrap();
          logsumexp(
            &mut alignfold_probs.context_profs_pair.0[(long_i, CONTEXT_INDEX_HAIRPIN)],
            y,
          );
        }
      }
      for (x, &y) in &unpair_probs_range_hairpin.1 {
        for i in range_inclusive(x.0, x.1) {
          let long_i = i.to_usize().unwrap();
          logsumexp(
            &mut alignfold_probs.context_profs_pair.1[(long_i, CONTEXT_INDEX_HAIRPIN)],
            y,
          );
        }
      }
      alignfold_probs
        .context_profs_pair
        .0
        .slice_mut(s![.., ..CONTEXT_INDEX_MULTIBRANCH])
        .mapv_inplace(expf);
      let fold = 1.
        - alignfold_probs
          .context_profs_pair
          .0
          .slice_mut(s![.., ..CONTEXT_INDEX_MULTIBRANCH])
          .sum_axis(Axis(1));
      alignfold_probs
        .context_profs_pair
        .0
        .slice_mut(s![.., CONTEXT_INDEX_MULTIBRANCH])
        .assign(&fold);
      alignfold_probs
        .context_profs_pair
        .1
        .slice_mut(s![.., ..CONTEXT_INDEX_MULTIBRANCH])
        .mapv_inplace(expf);
      let fold = 1.
        - alignfold_probs
          .context_profs_pair
          .1
          .slice_mut(s![.., ..CONTEXT_INDEX_MULTIBRANCH])
          .sum_axis(Axis(1));
      alignfold_probs
        .context_profs_pair
        .1
        .slice_mut(s![.., CONTEXT_INDEX_MULTIBRANCH])
        .assign(&fold);
    }
    if produces_match_probs {
      for (x, y) in alignfold_probs.loopmatch_probs.iter_mut() {
        match alignfold_probs.match_probs.get_mut(x) {
          Some(x) => {
            logsumexp(x, *y);
            *x = expf(*x);
          }
          None => {
            alignfold_probs.match_probs.insert(*x, expf(*y));
          }
        }
        *y = expf(*y);
      }
      for x in alignfold_probs.pairmatch_probs.values_mut() {
        *x = expf(*x);
      }
    }
    if trains_alignfold_scores {
      for x in alignfold_counts_expected.hairpin_scores_len.iter_mut() {
        *x = expf(*x);
      }
      for x in alignfold_counts_expected.bulge_scores_len.iter_mut() {
        *x = expf(*x);
      }
      for x in alignfold_counts_expected.interior_scores_len.iter_mut() {
        *x = expf(*x);
      }
      for x in alignfold_counts_expected
        .interior_scores_symmetric
        .iter_mut()
      {
        *x = expf(*x);
      }
      for x in alignfold_counts_expected
        .interior_scores_asymmetric
        .iter_mut()
      {
        *x = expf(*x);
      }
      for x in alignfold_counts_expected.stack_scores.iter_mut() {
        for x in x.iter_mut() {
          for x in x.iter_mut() {
            for x in x.iter_mut() {
              *x = expf(*x);
            }
          }
        }
      }
      for x in alignfold_counts_expected
        .terminal_mismatch_scores
        .iter_mut()
      {
        for x in x.iter_mut() {
          for x in x.iter_mut() {
            for x in x.iter_mut() {
              *x = expf(*x);
            }
          }
        }
      }
      for x in alignfold_counts_expected.dangling_scores_left.iter_mut() {
        for x in x.iter_mut() {
          for x in x.iter_mut() {
            *x = expf(*x);
          }
        }
      }
      for x in alignfold_counts_expected.dangling_scores_right.iter_mut() {
        for x in x.iter_mut() {
          for x in x.iter_mut() {
            *x = expf(*x);
          }
        }
      }
      for x in alignfold_counts_expected
        .interior_scores_explicit
        .iter_mut()
      {
        for x in x.iter_mut() {
          *x = expf(*x);
        }
      }
      for x in alignfold_counts_expected.bulge_scores_0x1.iter_mut() {
        *x = expf(*x);
      }
      for x in alignfold_counts_expected.interior_scores_1x1.iter_mut() {
        for x in x.iter_mut() {
          *x = expf(*x);
        }
      }
      for x in alignfold_counts_expected.helix_close_scores.iter_mut() {
        for x in x.iter_mut() {
          *x = expf(*x);
        }
      }
      for x in alignfold_counts_expected.basepair_scores.iter_mut() {
        for x in x.iter_mut() {
          *x = expf(*x);
        }
      }
      alignfold_counts_expected.multibranch_score_base =
        expf(alignfold_counts_expected.multibranch_score_base);
      alignfold_counts_expected.multibranch_score_basepair =
        expf(alignfold_counts_expected.multibranch_score_basepair);
      alignfold_counts_expected.multibranch_score_unpair =
        expf(alignfold_counts_expected.multibranch_score_unpair);
      alignfold_counts_expected.external_score_basepair =
        expf(alignfold_counts_expected.external_score_basepair);
      alignfold_counts_expected.external_score_unpair =
        expf(alignfold_counts_expected.external_score_unpair);
      alignfold_counts_expected.match2match_score =
        expf(alignfold_counts_expected.match2match_score);
      alignfold_counts_expected.match2insert_score =
        expf(alignfold_counts_expected.match2insert_score);
      alignfold_counts_expected.insert_extend_score =
        expf(alignfold_counts_expected.insert_extend_score);
      alignfold_counts_expected.init_match_score = expf(alignfold_counts_expected.init_match_score);
      alignfold_counts_expected.init_insert_score =
        expf(alignfold_counts_expected.init_insert_score);
      for x in alignfold_counts_expected.insert_scores.iter_mut() {
        *x = expf(*x);
      }
      for x in alignfold_counts_expected.match_scores.iter_mut() {
        for x in x.iter_mut() {
          *x = expf(*x);
        }
      }
    }
  }
  alignfold_probs
}

pub fn get_diff(x: usize, y: usize) -> usize {
  max(x, y) - min(x, y)
}

pub fn get_hairpin_score(x: &AlignfoldScores, y: SeqSlice, z: &(usize, usize)) -> Score {
  let a = z.1 - z.0 - 1;
  x.hairpin_scores_len_cumulative[a] + get_junction_score_single(x, y, z)
}

pub fn get_twoloop_score(
  x: &AlignfoldScores,
  y: SeqSlice,
  z: &(usize, usize),
  a: &(usize, usize),
) -> Score {
  let b = (y[a.0], y[a.1]);
  let c = if z.0 + 1 == a.0 && z.1 - 1 == a.1 {
    get_stack_score(x, y, z, a)
  } else if z.0 + 1 == a.0 || z.1 - 1 == a.1 {
    get_bulge_score(x, y, z, a)
  } else {
    get_interior_score(x, y, z, a)
  };
  c + x.basepair_scores[b.0][b.1]
}

pub fn get_stack_score(
  x: &AlignfoldScores,
  y: SeqSlice,
  z: &(usize, usize),
  a: &(usize, usize),
) -> Score {
  let b = (y[z.0], y[z.1]);
  let c = (y[a.0], y[a.1]);
  x.stack_scores[b.0][b.1][c.0][c.1]
}

pub fn get_bulge_score(
  x: &AlignfoldScores,
  y: SeqSlice,
  z: &(usize, usize),
  a: &(usize, usize),
) -> Score {
  let b = a.0 - z.0 + z.1 - a.1 - 2;
  let c = if b == 1 {
    x.bulge_scores_0x1[if a.0 - z.0 - 1 == 1 {
      y[z.0 + 1]
    } else {
      y[z.1 - 1]
    }]
  } else {
    0.
  };
  c + x.bulge_scores_len_cumulative[b - 1]
    + get_junction_score_single(x, y, z)
    + get_junction_score_single(x, y, &(a.1, a.0))
}

pub fn get_interior_score(
  x: &AlignfoldScores,
  y: SeqSlice,
  z: &(usize, usize),
  a: &(usize, usize),
) -> Score {
  let b = (a.0 - z.0 - 1, z.1 - a.1 - 1);
  let c = b.0 + b.1;
  let d = if b.0 == b.1 {
    let d = if c == 2 {
      x.interior_scores_1x1[y[z.0 + 1]][y[z.1 - 1]]
    } else {
      0.
    };
    d + x.interior_scores_symmetric_cumulative[b.0 - 1]
  } else {
    x.interior_scores_asymmetric_cumulative[get_abs_diff(b.0, b.1) - 1]
  };
  let e = if b.0 <= 4 && b.1 <= 4 {
    x.interior_scores_explicit[b.0 - 1][b.1 - 1]
  } else {
    0.
  };
  d + e
    + x.interior_scores_len_cumulative[c - 2]
    + get_junction_score_single(x, y, z)
    + get_junction_score_single(x, y, &(a.1, a.0))
}

pub fn get_junction_score_single(x: &AlignfoldScores, y: SeqSlice, z: &(usize, usize)) -> Score {
  let a = (y[z.0], y[z.1]);
  get_helix_close_score(x, &a) + get_terminal_mismatch_score(x, &a, &(y[z.0 + 1], y[z.1 - 1]))
}

pub fn get_helix_close_score(x: &AlignfoldScores, y: &Basepair) -> Score {
  x.helix_close_scores[y.0][y.1]
}

pub fn get_terminal_mismatch_score(x: &AlignfoldScores, y: &Basepair, z: &Basepair) -> Score {
  x.terminal_mismatch_scores[y.0][y.1][z.0][z.1]
}

pub fn get_junction_score(x: &AlignfoldScores, y: SeqSlice, z: &(usize, usize)) -> Score {
  let a = (y[z.0], y[z.1]);
  let b = 1;
  let c = y.len() - 2;
  get_helix_close_score(x, &a)
    + if z.0 < c {
      x.dangling_scores_left[a.0][a.1][y[z.0 + 1]]
    } else {
      0.
    }
    + if z.1 > b {
      x.dangling_scores_right[a.0][a.1][y[z.1 - 1]]
    } else {
      0.
    }
}

pub fn get_dict_min_stack(x: &Basepair, y: &Basepair) -> (Basepair, Basepair) {
  let z = (*x, *y);
  let a = ((y.1, y.0), (x.1, x.0));
  if z < a {
    z
  } else {
    a
  }
}

pub fn get_dict_min_pair(x: &(usize, usize)) -> (usize, usize) {
  let y = (x.1, x.0);
  if *x < y {
    *x
  } else {
    y
  }
}

pub fn get_num_unpairs_multibranch(
  x: &(usize, usize),
  y: &Vec<(usize, usize)>,
  z: SeqSlice,
) -> usize {
  let mut a = 0;
  let mut b = (0, 0);
  for y in y {
    let c = if b == (0, 0) { x.0 + 1 } else { b.1 + 1 };
    for &d in &z[c..y.0] {
      a += usize::from(d != PSEUDO_BASE);
    }
    b = *y;
  }
  for &c in &z[b.1 + 1..x.1] {
    a += usize::from(c != PSEUDO_BASE);
  }
  a
}

pub fn get_num_unpairs_external(x: &Vec<(usize, usize)>, y: SeqSlice) -> usize {
  let mut z = 0;
  let mut a = (0, 0);
  for x in x {
    let b = if a == (0, 0) { 0 } else { a.1 + 1 };
    for &b in y[b..x.0].iter() {
      z += usize::from(b != PSEUDO_BASE);
    }
    a = *x;
  }
  for &b in y[a.1 + 1..y.len()].iter() {
    z += usize::from(b != PSEUDO_BASE);
  }
  z
}

pub fn consprob_trained<T>(
  thread_pool: &mut Pool,
  seqs: &SeqSlices,
  min_basepair_prob: Prob,
  min_match_prob: Prob,
  produces_struct_profs: bool,
  produces_match_probs: bool,
  train_type: TrainType,
) -> (ProbMatSetsAvg<T>, MatchProbsHashedIds<T>)
where
  T: HashIndex,
{
  let trained = AlignfoldScores::load_trained_scores();
  let mut align_scores = AlignScores::new(0.);
  copy_alignfold_scores_align(&mut align_scores, &trained);
  let ref_align_scores = &align_scores;
  let mut fold_scores = FoldScoreSets::new(0.);
  copy_alignfold_scores_fold(&mut fold_scores, &trained);
  let ref_fold_scores = &fold_scores;
  let alignfold_scores = if matches!(train_type, TrainType::TrainedTransfer) {
    trained
  } else if matches!(train_type, TrainType::TrainedRandinit) {
    AlignfoldScores::load_trained_scores_randinit()
  } else {
    let mut transferred = AlignfoldScores::new(0.);
    transferred.transfer();
    transferred
  };
  let num_seqs = seqs.len();
  let mut basepair_prob_mats = vec![SparseProbMat::<T>::new(); num_seqs];
  let mut max_basepair_spans = vec![T::zero(); num_seqs];
  let mut fold_score_sets = vec![FoldScoresTrained::<T>::new(); num_seqs];
  let ref_alignfold_scores = &alignfold_scores;
  let uses_contra_model = true;
  let allows_short_hairpins = false;
  thread_pool.scoped(|scope| {
    for (x, y, z, a) in multizip((
      basepair_prob_mats.iter_mut(),
      max_basepair_spans.iter_mut(),
      seqs.iter(),
      fold_score_sets.iter_mut(),
    )) {
      let b = z.len();
      scope.execute(move || {
        let c = mccaskill_algo(
          &z[1..b - 1],
          uses_contra_model,
          allows_short_hairpins,
          ref_fold_scores,
        )
        .0;
        *x = filter_basepair_probs::<T>(&c, min_basepair_prob);
        *y = get_max_basepair_span::<T>(x);
        *a = FoldScoresTrained::<T>::set_curr_scores(ref_alignfold_scores, z, x);
      });
    }
  });
  let mut alignfold_probs_hashed_ids = AlignfoldProbsHashedIds::<T>::default();
  let mut match_probs_hashed_ids = SparseProbsHashedIds::<T>::default();
  for x in 0..num_seqs {
    for y in x + 1..num_seqs {
      let y = (x, y);
      alignfold_probs_hashed_ids.insert(y, AlignfoldProbMats::<T>::origin());
      match_probs_hashed_ids.insert(y, SparseProbMat::<T>::default());
    }
  }
  thread_pool.scoped(|x| {
    for (y, z) in match_probs_hashed_ids.iter_mut() {
      let y = (seqs[y.0], seqs[y.1]);
      x.execute(move || {
        *z = filter_match_probs(&durbin_algo(&y, ref_align_scores), min_match_prob);
      });
    }
  });
  let trains_alignfold_scores = false;
  thread_pool.scoped(|x| {
    let alignfold_scores = &alignfold_scores;
    for (y, z) in alignfold_probs_hashed_ids.iter_mut() {
      let seq_pair = (seqs[y.0], seqs[y.1]);
      let seq_len_pair = (
        T::from_usize(seq_pair.0.len()).unwrap(),
        T::from_usize(seq_pair.1.len()).unwrap(),
      );
      let max_basepair_span_pair = (max_basepair_spans[y.0], max_basepair_spans[y.1]);
      let basepair_probs_pair = (&basepair_prob_mats[y.0], &basepair_prob_mats[y.1]);
      let fold_scores_pair = (&fold_score_sets[y.0], &fold_score_sets[y.1]);
      let match_probs = &match_probs_hashed_ids[y];
      let (
        forward_pos_pairs,
        backward_pos_pairs,
        _,
        pos_quads_hashed_lens,
        matchable_poss,
        matchable_poss2,
      ) = get_sparse_poss(&basepair_probs_pair, match_probs, &seq_len_pair);
      x.execute(move || {
        *z = consprob_core::<T>((
          &seq_pair,
          alignfold_scores,
          &max_basepair_span_pair,
          match_probs,
          produces_struct_profs,
          trains_alignfold_scores,
          &mut AlignfoldScores::new(NEG_INFINITY),
          &forward_pos_pairs,
          &backward_pos_pairs,
          &pos_quads_hashed_lens,
          &fold_scores_pair,
          produces_match_probs,
          &matchable_poss,
          &matchable_poss2,
        ))
        .0;
      });
    }
  });
  let mut alignfold_prob_mats_avg = vec![AlignfoldProbMatsAvg::<T>::origin(); num_seqs];
  thread_pool.scoped(|x| {
    let y = &alignfold_probs_hashed_ids;
    for (z, a) in alignfold_prob_mats_avg.iter_mut().enumerate() {
      let b = seqs[z].len();
      x.execute(move || {
        *a = pair_probs2avg_probs::<T>(y, z, num_seqs, b, produces_struct_profs);
      });
    }
  });
  let mut match_probs_hashed_ids = MatchProbsHashedIds::<T>::default();
  if produces_match_probs {
    for x in 0..num_seqs {
      for y in x + 1..num_seqs {
        let y = (x, y);
        let z = &alignfold_probs_hashed_ids[&y];
        let mut a = MatchProbMats::<T>::new();
        a.loopmatch_probs = z.loopmatch_probs.clone();
        a.pairmatch_probs = z.pairmatch_probs.clone();
        a.match_probs = z.match_probs.clone();
        match_probs_hashed_ids.insert(y, a);
      }
    }
  }
  (alignfold_prob_mats_avg, match_probs_hashed_ids)
}

pub fn constrain<T>(
  thread_pool: &mut Pool,
  train_data: &mut TrainData<T>,
  output_file_path: &Path,
  enables_randinit: bool,
  learning_tolerance: Score,
) where
  T: HashIndex,
{
  let mut alignfold_scores = AlignfoldScores::new(0.);
  if enables_randinit {
    alignfold_scores.rand_init();
  } else {
    alignfold_scores.transfer();
  }
  for x in train_data.iter_mut() {
    x.set_curr_scores(&alignfold_scores);
  }
  let mut old_alignfold_scores = alignfold_scores.clone();
  let mut old_cost = INFINITY;
  let mut old_accuracy = NEG_INFINITY;
  let mut costs = Probs::new();
  let mut accuracies = costs.clone();
  let mut epoch = 0;
  let mut regularizers = Regularizers::from(vec![1.; alignfold_scores.len()]);
  let num_data = train_data.len() as Score;
  let produces_struct_profs = false;
  let trains_alignfold_scores = true;
  let produces_match_probs = true;
  loop {
    thread_pool.scoped(|scope| {
      let alignfold_scores = &alignfold_scores;
      for train_datum in train_data.iter_mut() {
        train_datum.alignfold_counts_expected = AlignfoldScores::new(NEG_INFINITY);
        let seq_pair = (&train_datum.seq_pair.0[..], &train_datum.seq_pair.1[..]);
        let max_basepair_span_pair = &train_datum.max_basepair_span_pair;
        let alignfold_counts_expected = &mut train_datum.alignfold_counts_expected;
        let accuracy = &mut train_datum.accuracy;
        let global_sum = &mut train_datum.global_sum;
        let forward_pos_pairs = &train_datum.forward_pos_pairs;
        let backward_pos_pairs = &train_datum.backward_pos_pairs;
        let pos_quads_hashed_lens = &train_datum.pos_quads_hashed_lens;
        let matchable_poss = &train_datum.matchable_poss;
        let matchable_poss2 = &train_datum.matchable_poss2;
        let match_probs = &train_datum.match_probs;
        let alignfold = &train_datum.alignfold;
        let fold_scores_pair = (
          &train_datum.fold_scores_pair.0,
          &train_datum.fold_scores_pair.1,
        );
        scope.execute(move || {
          let x = consprob_core::<T>((
            &seq_pair,
            alignfold_scores,
            max_basepair_span_pair,
            match_probs,
            produces_struct_profs,
            trains_alignfold_scores,
            alignfold_counts_expected,
            forward_pos_pairs,
            backward_pos_pairs,
            pos_quads_hashed_lens,
            &fold_scores_pair,
            produces_match_probs,
            matchable_poss,
            matchable_poss2,
          ));
          *accuracy = get_accuracy_expected::<T>(&seq_pair, alignfold, &x.0.match_probs);
          *global_sum = x.1;
        });
      }
    });
    let accuracy = train_data.iter().map(|x| x.accuracy).sum::<Score>() / train_data.len() as Score;
    let accuracy_change = accuracy - old_accuracy;
    if accuracy_change <= learning_tolerance {
      alignfold_scores = old_alignfold_scores;
      println!(
        "Accuracy change {accuracy_change} is <= learning tolerance {learning_tolerance}; training finished"
      );
      break;
    }
    old_alignfold_scores = alignfold_scores.clone();
    alignfold_scores.update(train_data, &mut regularizers);
    for x in train_data.iter_mut() {
      x.set_curr_scores(&alignfold_scores);
    }
    let cost = alignfold_scores.get_cost(&train_data[..], &regularizers);
    let avg_cost_change = (cost - old_cost) / num_data;
    if avg_cost_change >= 0. {
      println!("Average cost change {avg_cost_change} is not negative; training finished");
      alignfold_scores = old_alignfold_scores;
      break;
    }
    costs.push(cost);
    accuracies.push(accuracy);
    println!("Epoch {} finished (current cost = {}, current accuracy = {}, average cost change = {}, accuracy change = {})", epoch + 1, cost, accuracy, avg_cost_change, accuracy_change);
    epoch += 1;
    old_cost = cost;
    old_accuracy = accuracy;
  }
  write_alignfold_scores_trained(&alignfold_scores, enables_randinit);
  write_logs(&costs, &accuracies, output_file_path);
}

pub fn gapped2ungapped(x: &Seq) -> Seq {
  x.iter().filter(|&&x| x != PSEUDO_BASE).copied().collect()
}

pub fn bytes2seq_gapped(x: &[u8]) -> Seq {
  let mut y = Seq::new();
  for &x in x {
    let x = convert_char(x);
    y.push(x);
  }
  y
}

pub fn convert_char(x: u8) -> Base {
  match x {
    A_LOWER | A_UPPER => A,
    C_LOWER | C_UPPER => C,
    G_LOWER | G_UPPER => G,
    U_LOWER | U_UPPER => U,
    _ => PSEUDO_BASE,
  }
}

pub fn get_mismatch_pair(x: SeqSlice, y: &(usize, usize), z: bool) -> (usize, usize) {
  let mut a = if z {
    (x[y.1], x[y.0])
  } else {
    (PSEUDO_BASE, PSEUDO_BASE)
  };
  if z {
    for &b in &x[y.0 + 1..y.1] {
      if b != PSEUDO_BASE {
        a.0 = b;
        break;
      }
    }
    for i in (y.0 + 1..y.1).rev() {
      let x = x[i];
      if x != PSEUDO_BASE {
        a.1 = x;
        break;
      }
    }
  } else {
    for i in (0..y.0).rev() {
      let x = x[i];
      if x != PSEUDO_BASE {
        a.0 = x;
        break;
      }
    }
    let b = x.len();
    for &x in &x[y.1 + 1..b] {
      if x != PSEUDO_BASE {
        a.1 = x;
        break;
      }
    }
  }
  a
}

pub fn get_hairpin_len(x: SeqSlice, y: &(usize, usize)) -> usize {
  let mut z = 0;
  for &x in &x[y.0 + 1..y.1] {
    if x == PSEUDO_BASE {
      continue;
    }
    z += 1;
  }
  z
}

pub fn get_2loop_len_pair(x: SeqSlice, y: &(usize, usize), z: &(usize, usize)) -> (usize, usize) {
  let mut a = (0, 0);
  for &x in &x[y.0 + 1..z.0] {
    if x == PSEUDO_BASE {
      continue;
    }
    a.0 += 1;
  }
  for &x in &x[z.1 + 1..y.1] {
    if x == PSEUDO_BASE {
      continue;
    }
    a.1 += 1;
  }
  a
}

pub fn vec2struct(source: &Scores, uses_cumulative_scores: bool) -> AlignfoldScores {
  let mut target = AlignfoldScores::new(0.);
  let mut offset = 0;
  let len = target.hairpin_scores_len.len();
  for i in 0..len {
    let x = source[offset + i];
    if uses_cumulative_scores {
      target.hairpin_scores_len_cumulative[i] = x;
    } else {
      target.hairpin_scores_len[i] = x;
    }
  }
  offset += len;
  let len = target.bulge_scores_len.len();
  for i in 0..len {
    let x = source[offset + i];
    if uses_cumulative_scores {
      target.bulge_scores_len_cumulative[i] = x;
    } else {
      target.bulge_scores_len[i] = x;
    }
  }
  offset += len;
  let len = target.interior_scores_len.len();
  for i in 0..len {
    let x = source[offset + i];
    if uses_cumulative_scores {
      target.interior_scores_len_cumulative[i] = x;
    } else {
      target.interior_scores_len[i] = x;
    }
  }
  offset += len;
  let len = target.interior_scores_symmetric.len();
  for i in 0..len {
    let x = source[offset + i];
    if uses_cumulative_scores {
      target.interior_scores_symmetric_cumulative[i] = x;
    } else {
      target.interior_scores_symmetric[i] = x;
    }
  }
  offset += len;
  let len = target.interior_scores_asymmetric.len();
  for i in 0..len {
    let x = source[offset + i];
    if uses_cumulative_scores {
      target.interior_scores_asymmetric_cumulative[i] = x;
    } else {
      target.interior_scores_asymmetric[i] = x;
    }
  }
  offset += len;
  let len = target.stack_scores.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      for k in 0..len {
        for l in 0..len {
          if !has_canonical_basepair(&(k, l)) {
            continue;
          }
          let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
          if ((i, j), (k, l)) != dict_min_stack {
            continue;
          }
          target.stack_scores[i][j][k][l] = source[offset];
          offset += 1;
        }
      }
    }
  }
  let len = target.terminal_mismatch_scores.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      for k in 0..len {
        for l in 0..len {
          target.terminal_mismatch_scores[i][j][k][l] = source[offset];
          offset += 1;
        }
      }
    }
  }
  let len = target.dangling_scores_left.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      for k in 0..len {
        target.dangling_scores_left[i][j][k] = source[offset];
        offset += 1;
      }
    }
  }
  let len = target.dangling_scores_right.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      for k in 0..len {
        target.dangling_scores_right[i][j][k] = source[offset];
        offset += 1;
      }
    }
  }
  let len = target.helix_close_scores.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      target.helix_close_scores[i][j] = source[offset];
      offset += 1;
    }
  }
  let len = target.basepair_scores.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      let dict_min_basepair = get_dict_min_pair(&(i, j));
      if (i, j) != dict_min_basepair {
        continue;
      }
      target.basepair_scores[i][j] = source[offset];
      offset += 1;
    }
  }
  let len = target.interior_scores_explicit.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_len_pair = get_dict_min_pair(&(i, j));
      if (i, j) != dict_min_len_pair {
        continue;
      }
      target.interior_scores_explicit[i][j] = source[offset];
      offset += 1;
    }
  }
  let len = target.bulge_scores_0x1.len();
  for i in 0..len {
    target.bulge_scores_0x1[i] = source[offset + i];
  }
  offset += len;
  let len = target.interior_scores_1x1.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_basepair = get_dict_min_pair(&(i, j));
      if (i, j) != dict_min_basepair {
        continue;
      }
      target.interior_scores_1x1[i][j] = source[offset];
      offset += 1;
    }
  }
  target.multibranch_score_base = source[offset];
  offset += 1;
  target.multibranch_score_basepair = source[offset];
  offset += 1;
  target.multibranch_score_unpair = source[offset];
  offset += 1;
  target.external_score_basepair = source[offset];
  offset += 1;
  target.external_score_unpair = source[offset];
  offset += 1;
  target.match2match_score = source[offset];
  offset += 1;
  target.match2insert_score = source[offset];
  offset += 1;
  target.init_match_score = source[offset];
  offset += 1;
  target.insert_extend_score = source[offset];
  offset += 1;
  target.init_insert_score = source[offset];
  offset += 1;
  let len = target.insert_scores.len();
  for i in 0..len {
    target.insert_scores[i] = source[offset + i];
  }
  offset += len;
  let len = target.match_scores.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_match = get_dict_min_pair(&(i, j));
      if (i, j) != dict_min_match {
        continue;
      }
      target.match_scores[i][j] = source[offset];
      offset += 1;
    }
  }
  assert!(offset == target.len());
  target
}

pub fn struct2vec(source: &AlignfoldScores, uses_cumulative_scores: bool) -> Scores {
  let mut target = vec![0.; source.len()];
  let mut offset = 0;
  let len = source.hairpin_scores_len.len();
  for i in 0..len {
    target[offset + i] = if uses_cumulative_scores {
      source.hairpin_scores_len_cumulative[i]
    } else {
      source.hairpin_scores_len[i]
    };
  }
  offset += len;
  let len = source.bulge_scores_len.len();
  for i in 0..len {
    target[offset + i] = if uses_cumulative_scores {
      source.bulge_scores_len_cumulative[i]
    } else {
      source.bulge_scores_len[i]
    };
  }
  offset += len;
  let len = source.interior_scores_len.len();
  for i in 0..len {
    target[offset + i] = if uses_cumulative_scores {
      source.interior_scores_len_cumulative[i]
    } else {
      source.interior_scores_len[i]
    };
  }
  offset += len;
  let len = source.interior_scores_symmetric.len();
  for i in 0..len {
    target[offset + i] = if uses_cumulative_scores {
      source.interior_scores_symmetric_cumulative[i]
    } else {
      source.interior_scores_symmetric[i]
    };
  }
  offset += len;
  let len = source.interior_scores_asymmetric.len();
  for i in 0..len {
    target[offset + i] = if uses_cumulative_scores {
      source.interior_scores_asymmetric_cumulative[i]
    } else {
      source.interior_scores_asymmetric[i]
    };
  }
  offset += len;
  let len = source.stack_scores.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      for k in 0..len {
        for l in 0..len {
          if !has_canonical_basepair(&(k, l)) {
            continue;
          }
          let dict_min_stack = get_dict_min_stack(&(i, j), &(k, l));
          if ((i, j), (k, l)) != dict_min_stack {
            continue;
          }
          target[offset] = source.stack_scores[i][j][k][l];
          offset += 1;
        }
      }
    }
  }
  let len = source.terminal_mismatch_scores.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      for k in 0..len {
        for l in 0..len {
          target[offset] = source.terminal_mismatch_scores[i][j][k][l];
          offset += 1;
        }
      }
    }
  }
  let len = source.dangling_scores_left.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      for k in 0..len {
        target[offset] = source.dangling_scores_left[i][j][k];
        offset += 1;
      }
    }
  }
  let len = source.dangling_scores_right.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      for k in 0..len {
        target[offset] = source.dangling_scores_right[i][j][k];
        offset += 1;
      }
    }
  }
  let len = source.helix_close_scores.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      target[offset] = source.helix_close_scores[i][j];
      offset += 1;
    }
  }
  let len = source.basepair_scores.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      let dict_min_basepair = get_dict_min_pair(&(i, j));
      if (i, j) != dict_min_basepair {
        continue;
      }
      target[offset] = source.basepair_scores[i][j];
      offset += 1;
    }
  }
  let len = source.interior_scores_explicit.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_len_pair = get_dict_min_pair(&(i, j));
      if (i, j) != dict_min_len_pair {
        continue;
      }
      target[offset] = source.interior_scores_explicit[i][j];
      offset += 1;
    }
  }
  let len = source.bulge_scores_0x1.len();
  for (i, &x) in source.bulge_scores_0x1.iter().enumerate() {
    target[offset + i] = x;
  }
  offset += len;
  let len = source.interior_scores_1x1.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_basepair = get_dict_min_pair(&(i, j));
      if (i, j) != dict_min_basepair {
        continue;
      }
      target[offset] = source.interior_scores_1x1[i][j];
      offset += 1;
    }
  }
  target[offset] = source.multibranch_score_base;
  offset += 1;
  target[offset] = source.multibranch_score_basepair;
  offset += 1;
  target[offset] = source.multibranch_score_unpair;
  offset += 1;
  target[offset] = source.external_score_basepair;
  offset += 1;
  target[offset] = source.external_score_unpair;
  offset += 1;
  target[offset] = source.match2match_score;
  offset += 1;
  target[offset] = source.match2insert_score;
  offset += 1;
  target[offset] = source.init_match_score;
  offset += 1;
  target[offset] = source.insert_extend_score;
  offset += 1;
  target[offset] = source.init_insert_score;
  offset += 1;
  let len = source.insert_scores.len();
  for (i, &x) in source.insert_scores.iter().enumerate() {
    target[offset + i] = x;
  }
  offset += len;
  let len = source.match_scores.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_match = get_dict_min_pair(&(i, j));
      if (i, j) != dict_min_match {
        continue;
      }
      target[offset] = source.match_scores[i][j];
      offset += 1;
    }
  }
  assert!(offset == source.len());
  Array::from(target)
}

pub fn scores2bfgs_scores(x: &Scores) -> BfgsScores {
  let x: Vec<BfgsScore> = x.to_vec().iter().map(|x| *x as BfgsScore).collect();
  BfgsScores::from(x)
}

pub fn bfgs_scores2scores(x: &BfgsScores) -> Scores {
  let x: Vec<Score> = x.to_vec().iter().map(|x| *x as Score).collect();
  Scores::from(x)
}

pub fn get_regularizer(x: usize, y: Score) -> Regularizer {
  (x as Score / 2. + GAMMA_DISTRO_ALPHA) / (y / 2. + GAMMA_DISTRO_BETA)
}

pub fn write_alignfold_scores_trained(alignfold_scores: &AlignfoldScores, enables_randinit: bool) {
  let mut writer = BufWriter::new(
    File::create(if enables_randinit {
      TRAINED_SCORES_FILE_RANDINIT
    } else {
      TRAINED_SCORES_FILE
    })
    .unwrap(),
  );
  let mut buf = format!("use AlignfoldScores;\nimpl AlignfoldScores {{\npub fn load_trained_scores{}() -> AlignfoldScores {{\nAlignfoldScores {{\nhairpin_scores_len: ", if enables_randinit {"_randinit"} else {""});
  buf.push_str(&format!(
    "{:?},\nbulge_scores_len: ",
    &alignfold_scores.hairpin_scores_len
  ));
  buf.push_str(&format!(
    "{:?},\ninterior_scores_len: ",
    &alignfold_scores.bulge_scores_len
  ));
  buf.push_str(&format!(
    "{:?},\ninterior_scores_symmetric: ",
    &alignfold_scores.interior_scores_len
  ));
  buf.push_str(&format!(
    "{:?},\ninterior_scores_asymmetric: ",
    &alignfold_scores.interior_scores_symmetric
  ));
  buf.push_str(&format!(
    "{:?},\nstack_scores: ",
    &alignfold_scores.interior_scores_asymmetric
  ));
  buf.push_str(&format!(
    "{:?},\nterminal_mismatch_scores: ",
    &alignfold_scores.stack_scores
  ));
  buf.push_str(&format!(
    "{:?},\ndangling_scores_left: ",
    &alignfold_scores.terminal_mismatch_scores
  ));
  buf.push_str(&format!(
    "{:?},\ndangling_scores_right: ",
    &alignfold_scores.dangling_scores_left
  ));
  buf.push_str(&format!(
    "{:?},\nhelix_close_scores: ",
    &alignfold_scores.dangling_scores_right
  ));
  buf.push_str(&format!(
    "{:?},\nbasepair_scores: ",
    &alignfold_scores.helix_close_scores
  ));
  buf.push_str(&format!(
    "{:?},\ninterior_scores_explicit: ",
    &alignfold_scores.basepair_scores
  ));
  buf.push_str(&format!(
    "{:?},\nbulge_scores_0x1: ",
    &alignfold_scores.interior_scores_explicit
  ));
  buf.push_str(&format!(
    "{:?},\ninterior_scores_1x1: ",
    &alignfold_scores.bulge_scores_0x1
  ));
  buf.push_str(&format!(
    "{:?},\nmultibranch_score_base: ",
    &alignfold_scores.interior_scores_1x1
  ));
  buf.push_str(&format!(
    "{:?},\nmultibranch_score_basepair: ",
    alignfold_scores.multibranch_score_base
  ));
  buf.push_str(&format!(
    "{:?},\nmultibranch_score_unpair: ",
    alignfold_scores.multibranch_score_basepair
  ));
  buf.push_str(&format!(
    "{:?},\nexternal_score_basepair: ",
    alignfold_scores.multibranch_score_unpair
  ));
  buf.push_str(&format!(
    "{:?},\nexternal_score_unpair: ",
    alignfold_scores.external_score_basepair
  ));
  buf.push_str(&format!(
    "{:?},\nmatch2match_score: ",
    alignfold_scores.external_score_unpair
  ));
  buf.push_str(&format!(
    "{:?},\nmatch2insert_score: ",
    alignfold_scores.match2match_score
  ));
  buf.push_str(&format!(
    "{:?},\ninsert_extend_score: ",
    alignfold_scores.match2insert_score
  ));
  buf.push_str(&format!(
    "{:?},\ninit_match_score: ",
    alignfold_scores.insert_extend_score
  ));
  buf.push_str(&format!(
    "{:?},\ninit_insert_score: ",
    alignfold_scores.init_match_score
  ));
  buf.push_str(&format!(
    "{:?},\ninsert_scores: ",
    alignfold_scores.init_insert_score
  ));
  buf.push_str(&format!(
    "{:?},\nmatch_scores: ",
    alignfold_scores.insert_scores
  ));
  buf.push_str(&format!(
    "{:?},\nhairpin_scores_len_cumulative: ",
    alignfold_scores.match_scores
  ));
  buf.push_str(&format!(
    "{:?},\nbulge_scores_len_cumulative: ",
    &alignfold_scores.hairpin_scores_len_cumulative
  ));
  buf.push_str(&format!(
    "{:?},\ninterior_scores_len_cumulative: ",
    &alignfold_scores.bulge_scores_len_cumulative
  ));
  buf.push_str(&format!(
    "{:?},\ninterior_scores_symmetric_cumulative: ",
    &alignfold_scores.interior_scores_len_cumulative
  ));
  buf.push_str(&format!(
    "{:?},\ninterior_scores_asymmetric_cumulative: ",
    &alignfold_scores.interior_scores_symmetric_cumulative
  ));
  buf.push_str(&format!(
    "{:?},",
    &alignfold_scores.interior_scores_asymmetric_cumulative
  ));
  buf.push_str(&String::from("\n}\n}\n}"));
  let _ = writer.write_all(buf.as_bytes());
}

pub fn write_logs(x: &Probs, y: &Probs, z: &Path) {
  let mut z = BufWriter::new(File::create(z).unwrap());
  let mut a = String::new();
  for (x, y) in x.iter().zip(y.iter()) {
    a.push_str(&format!("{x},{y}\n"));
  }
  let _ = z.write_all(a.as_bytes());
}

pub fn copy_alignfold_scores_align(x: &mut AlignScores, y: &AlignfoldScores) {
  x.match2match_score = y.match2match_score;
  x.match2insert_score = y.match2insert_score;
  x.insert_extend_score = y.insert_extend_score;
  x.init_match_score = y.init_match_score;
  x.init_insert_score = y.init_insert_score;
  let len = x.insert_scores.len();
  for i in 0..len {
    x.insert_scores[i] = y.insert_scores[i];
  }
  let len = x.match_scores.len();
  for i in 0..len {
    for j in 0..len {
      x.match_scores[i][j] = y.match_scores[i][j];
    }
  }
}

pub fn copy_alignfold_scores_fold(
  fold_scores: &mut FoldScoreSets,
  alignfold_scores: &AlignfoldScores,
) {
  for (x, &y) in fold_scores
    .hairpin_scores_len
    .iter_mut()
    .zip(alignfold_scores.hairpin_scores_len.iter())
  {
    *x = y;
  }
  for (x, &y) in fold_scores
    .bulge_scores_len
    .iter_mut()
    .zip(alignfold_scores.bulge_scores_len.iter())
  {
    *x = y;
  }
  for (x, &y) in fold_scores
    .interior_scores_len
    .iter_mut()
    .zip(alignfold_scores.interior_scores_len.iter())
  {
    *x = y;
  }
  for (x, &y) in fold_scores
    .interior_scores_symmetric
    .iter_mut()
    .zip(alignfold_scores.interior_scores_symmetric.iter())
  {
    *x = y;
  }
  for (x, &y) in fold_scores
    .interior_scores_asymmetric
    .iter_mut()
    .zip(alignfold_scores.interior_scores_asymmetric.iter())
  {
    *x = y;
  }
  let len = fold_scores.stack_scores.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      for k in 0..len {
        for l in 0..len {
          if !has_canonical_basepair(&(k, l)) {
            continue;
          }
          fold_scores.stack_scores[i][j][k][l] = alignfold_scores.stack_scores[i][j][k][l];
        }
      }
    }
  }
  let len = fold_scores.terminal_mismatch_scores.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      for k in 0..len {
        for l in 0..len {
          fold_scores.terminal_mismatch_scores[i][j][k][l] =
            alignfold_scores.terminal_mismatch_scores[i][j][k][l];
        }
      }
    }
  }
  let len = fold_scores.dangling_scores_left.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      for k in 0..len {
        fold_scores.dangling_scores_left[i][j][k] = alignfold_scores.dangling_scores_left[i][j][k];
      }
    }
  }
  let len = fold_scores.dangling_scores_right.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      for k in 0..len {
        fold_scores.dangling_scores_right[i][j][k] =
          alignfold_scores.dangling_scores_right[i][j][k];
      }
    }
  }
  let len = fold_scores.helix_close_scores.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      fold_scores.helix_close_scores[i][j] = alignfold_scores.helix_close_scores[i][j];
    }
  }
  let len = fold_scores.basepair_scores.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      fold_scores.basepair_scores[i][j] = alignfold_scores.basepair_scores[i][j];
    }
  }
  let len = fold_scores.interior_scores_explicit.len();
  for i in 0..len {
    for j in 0..len {
      fold_scores.interior_scores_explicit[i][j] = alignfold_scores.interior_scores_explicit[i][j];
    }
  }
  for (x, &y) in fold_scores
    .bulge_scores_0x1
    .iter_mut()
    .zip(alignfold_scores.bulge_scores_0x1.iter())
  {
    *x = y;
  }
  let len = fold_scores.interior_scores_1x1.len();
  for i in 0..len {
    for j in 0..len {
      fold_scores.interior_scores_1x1[i][j] = alignfold_scores.interior_scores_1x1[i][j];
    }
  }
  fold_scores.multibranch_score_base = alignfold_scores.multibranch_score_base;
  fold_scores.multibranch_score_basepair = alignfold_scores.multibranch_score_basepair;
  fold_scores.multibranch_score_unpair = alignfold_scores.multibranch_score_unpair;
  fold_scores.external_score_basepair = alignfold_scores.external_score_basepair;
  fold_scores.external_score_unpair = alignfold_scores.external_score_unpair;
  fold_scores.accumulate();
}

pub fn print_train_info(alignfold_scores: &AlignfoldScores) {
  let mut num_groups = 0;
  println!("Training the parameter groups below");
  println!("-----------------------------------");
  println!("Groups from the CONTRAfold model...");
  println!(
    "Hairpin loop length (group size {})",
    alignfold_scores.hairpin_scores_len.len()
  );
  num_groups += 1;
  println!(
    "Bulge loop length (group size {})",
    alignfold_scores.bulge_scores_len.len()
  );
  num_groups += 1;
  println!(
    "Interior loop length (group size {})",
    alignfold_scores.interior_scores_len.len()
  );
  num_groups += 1;
  println!(
    "Interior loop length symmetric (group size {})",
    alignfold_scores.interior_scores_symmetric.len()
  );
  num_groups += 1;
  println!(
    "Interior loop length asymmetric (group size {})",
    alignfold_scores.interior_scores_asymmetric.len()
  );
  num_groups += 1;
  let mut group_size = 0;
  let len = alignfold_scores.stack_scores.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      for k in 0..len {
        for l in 0..len {
          if !has_canonical_basepair(&(k, l)) {
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
  println!("Stacking (group size {group_size})");
  num_groups += 1;
  let mut group_size = 0;
  let len = alignfold_scores.terminal_mismatch_scores.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      for _ in 0..len {
        for _ in 0..len {
          group_size += 1;
        }
      }
    }
  }
  println!("Terminal mismatch (group size {group_size})");
  num_groups += 1;
  let mut group_size = 0;
  let len = alignfold_scores.dangling_scores_left.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      for _ in 0..len {
        group_size += 1;
      }
    }
  }
  let len = alignfold_scores.dangling_scores_right.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      for _ in 0..len {
        group_size += 1;
      }
    }
  }
  println!("Dangling (group size {group_size})");
  num_groups += 1;
  let mut group_size = 0;
  let len = alignfold_scores.helix_close_scores.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      group_size += 1;
    }
  }
  println!("Helix end (group size {group_size})");
  num_groups += 1;
  let mut group_size = 0;
  let len = alignfold_scores.basepair_scores.len();
  for i in 0..len {
    for j in 0..len {
      if !has_canonical_basepair(&(i, j)) {
        continue;
      }
      let dict_min_basepair = get_dict_min_pair(&(i, j));
      if (i, j) != dict_min_basepair {
        continue;
      }
      group_size += 1;
    }
  }
  println!("Base-pairing (group size {group_size})");
  num_groups += 1;
  let mut group_size = 0;
  let len = alignfold_scores.interior_scores_explicit.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_len_pair = get_dict_min_pair(&(i, j));
      if (i, j) != dict_min_len_pair {
        continue;
      }
      group_size += 1;
    }
  }
  println!("Interior loop length explicit (group size {group_size})");
  num_groups += 1;
  println!(
    "Bulge loop length 0x1 (group size {})",
    alignfold_scores.bulge_scores_0x1.len()
  );
  num_groups += 1;
  let mut group_size = 0;
  let len = alignfold_scores.interior_scores_1x1.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_basepair = get_dict_min_pair(&(i, j));
      if (i, j) != dict_min_basepair {
        continue;
      }
      group_size += 1;
    }
  }
  println!("Interior loop length 1x1 (group size {group_size})");
  num_groups += 1;
  println!("Multi-loop length (group size {GROUP_SIZE_MULTIBRANCH})");
  num_groups += 1;
  println!("External-loop length (group size {GROUP_SIZE_EXTERNAL})");
  num_groups += 1;
  println!("-----------------------------------");
  println!("Groups from the CONTRAlign model...");
  println!("Match transition (group size {GROUP_SIZE_MATCH_TRANSITION})");
  num_groups += 1;
  println!("Insert transition (group size {GROUP_SIZE_INSERT_TRANSITION})");
  num_groups += 1;
  println!(
    "Insert emission (group size {})",
    alignfold_scores.insert_scores.len()
  );
  num_groups += 1;
  let mut group_size = 0;
  let len = alignfold_scores.match_scores.len();
  for i in 0..len {
    for j in 0..len {
      let dict_min_match = get_dict_min_pair(&(i, j));
      if (i, j) != dict_min_match {
        continue;
      }
      group_size += 1;
    }
  }
  println!("Match emission (group size {group_size})");
  num_groups += 1;
  println!("-----------------------------------");
  println!(
    "Total # scoring parameters (from {} groups) to be trained: {}",
    num_groups,
    alignfold_scores.len()
  );
}
