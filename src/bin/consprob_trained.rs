extern crate consprob_trained;

use consprob_trained::*;
use std::env;

fn main() {
  let args = env::args().collect::<Args>();
  let program_name = args[0].clone();
  let mut opts = Options::new();
  opts.reqopt(
    "i",
    "input_file_path",
    "An input FASTA file path containing RNA sequences to predict probabilities",
    "STR",
  );
  opts.reqopt("o", "output_dir_path", "An output directory path", "STR");
  opts.optopt(
    "",
    "min_basepair_prob",
    &format!("A minimum base-pairing probability (Use {DEFAULT_MIN_BASEPAIR_PROB} by default)"),
    "FLOAT",
  );
  opts.optopt(
    "",
    "min_match_prob",
    &format!("A minimum matching probability (Use {DEFAULT_MIN_MATCH_PROB} by default)"),
    "FLOAT",
  );
  opts.optopt("u", "train_type", &format!("Choose a scoring parameter training type from trained_transfer, trained_random_init, transferred_only (Use {DEFAULT_TRAIN_TYPE} by default)"), "STR");
  opts.optopt(
    "t",
    "num_threads",
    "The number of threads in multithreading (Use all the threads of this computer by default)",
    "UINT",
  );
  opts.optflag(
    "s",
    "produces_struct_profs",
    "Also compute RNA structural context profiles",
  );
  opts.optflag(
    "m",
    "produces_match_probs",
    "Also compute nucleotide matching probabilities",
  );
  opts.optflag("h", "help", "Print a help menu");
  let matches = match opts.parse(&args[1..]) {
    Ok(x) => x,
    Err(x) => {
      print_program_usage(&program_name, &opts);
      panic!("{}", x.to_string())
    }
  };
  if matches.opt_present("h") {
    print_program_usage(&program_name, &opts);
    return;
  }
  let input_file_path = matches.opt_str("i").unwrap();
  let input_file_path = Path::new(&input_file_path);
  let min_basepair_prob = if matches.opt_present("min_basepair_prob") {
    matches
      .opt_str("min_basepair_prob")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_MIN_BASEPAIR_PROB
  };
  let min_match_prob = if matches.opt_present("min_match_prob") {
    matches.opt_str("min_match_prob").unwrap().parse().unwrap()
  } else {
    DEFAULT_MIN_MATCH_PROB
  };
  let train_type = if matches.opt_present("u") {
    let train_type_str = matches.opt_str("u").unwrap();
    if train_type_str == "trained_transfer" {
      TrainType::TrainedTransfer
    } else if train_type_str == "trained_random_init" {
      TrainType::TrainedRandinit
    } else if train_type_str == "transferred_only" {
      TrainType::TransferredOnly
    } else {
      panic!();
    }
  } else {
    TrainType::TrainedTransfer
  };
  let num_threads = if matches.opt_present("t") {
    matches.opt_str("t").unwrap().parse().unwrap()
  } else {
    num_cpus::get() as NumThreads
  };
  let produces_struct_profs = matches.opt_present("s");
  let produces_match_probs = matches.opt_present("m");
  let output_dir_path = matches.opt_str("o").unwrap();
  let output_dir_path = Path::new(&output_dir_path);
  let fasta_file_reader = Reader::from_file(Path::new(&input_file_path)).unwrap();
  let mut fasta_records = FastaRecords::new();
  let mut max_seq_len = 0;
  for x in fasta_file_reader.records() {
    let x = x.unwrap();
    let mut y = bytes2seq(x.seq());
    y.insert(0, PSEUDO_BASE);
    y.push(PSEUDO_BASE);
    let z = y.len();
    if z > max_seq_len {
      max_seq_len = z;
    }
    fasta_records.push(FastaRecord::new(String::from(x.id()), y));
  }
  let mut thread_pool = Pool::new(num_threads);
  let seqs = fasta_records.iter().map(|x| &x.seq[..]).collect();
  if max_seq_len <= u8::MAX as usize {
    let (alignfold_prob_mats_avg, match_probs_hashed_ids) = consprob_trained::<u8>(
      &mut thread_pool,
      &seqs,
      min_basepair_prob,
      min_match_prob,
      produces_struct_profs,
      produces_match_probs,
      train_type,
    );
    write_alignfold_prob_mats(
      output_dir_path,
      &alignfold_prob_mats_avg,
      &match_probs_hashed_ids,
      produces_struct_profs,
      produces_match_probs,
    );
  } else {
    let (alignfold_prob_mats_avg, match_probs_hashed_ids) = consprob_trained::<u16>(
      &mut thread_pool,
      &seqs,
      min_basepair_prob,
      min_match_prob,
      produces_struct_profs,
      produces_match_probs,
      train_type,
    );
    write_alignfold_prob_mats(
      output_dir_path,
      &alignfold_prob_mats_avg,
      &match_probs_hashed_ids,
      produces_struct_profs,
      produces_match_probs,
    );
  }
  write_readme(output_dir_path, &String::from(README_CONTENTS));
}
