extern crate consprob_trained;

use consprob_trained::*;
use std::env;

fn main() {
  let args = env::args().collect::<Args>();
  let program_name = args[0].clone();
  let mut opts = Options::new();
  opts.reqopt(
    "i",
    "input_dir",
    "An input directory path containing RNA structural alignments in FASTA format",
    "STR",
  );
  opts.reqopt(
    "o",
    "output_file_path",
    "An output file path to record intermediate training logs",
    "STR",
  );
  opts.optopt(
    "",
    "min_base_pair_prob",
    &format!("A minimum base-pairing probability (Use {DEFAULT_MIN_BPP_TRAIN} by default)"),
    "FLOAT",
  );
  opts.optopt(
    "",
    "min_align_prob",
    &format!("A minimum aligning probability (Use {DEFAULT_MIN_ALIGN_PROB_TRAIN} by default)"),
    "FLOAT",
  );
  opts.optopt(
    "",
    "learning_tolerance",
    &format!(
      "The lower threshold of training accuracy change to quit learning (Use {DEFAULT_LEARNING_TOLERANCE} by default)"
    ),
    "FLOAT",
  );
  opts.optopt(
    "t",
    "num_of_threads",
    "The number of threads in multithreading (Use all the threads of this computer by default)",
    "UINT",
  );
  opts.optflag(
    "r",
    "enable_random_init",
    "Enable the random initialization of alignment scoring parameters to be trained",
  );
  opts.optflag("h", "help", "Print a help menu");
  let matches = match opts.parse(&args[1..]) {
    Ok(opt) => opt,
    Err(failure) => {
      print_program_usage(&program_name, &opts);
      panic!("{}", failure.to_string())
    }
  };
  if matches.opt_present("h") {
    print_program_usage(&program_name, &opts);
    return;
  }
  let input_dir_path = matches.opt_str("i").unwrap();
  let input_dir_path = Path::new(&input_dir_path);
  let min_bpp = if matches.opt_present("min_base_pair_prob") {
    matches
      .opt_str("min_base_pair_prob")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_MIN_BPP_TRAIN
  };
  let min_align_prob = if matches.opt_present("min_align_prob") {
    matches.opt_str("min_align_prob").unwrap().parse().unwrap()
  } else {
    DEFAULT_MIN_ALIGN_PROB_TRAIN
  };
  let learning_tolerance = if matches.opt_present("learning_tolerance") {
    matches
      .opt_str("learning_tolerance")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_LEARNING_TOLERANCE
  };
  let enable_random_init = matches.opt_present("r");
  let num_of_threads = if matches.opt_present("t") {
    matches.opt_str("t").unwrap().parse().unwrap()
  } else {
    num_cpus::get() as NumOfThreads
  };
  println!("# threads = {num_of_threads}");
  let output_file_path = matches.opt_str("o").unwrap();
  let output_file_path = Path::new(&output_file_path);
  print_train_info(&FeatureCountSets::new(0.));
  let entries: Vec<DirEntry> = read_dir(input_dir_path)
    .unwrap()
    .map(|x| x.unwrap())
    .collect();
  let num_of_entries = entries.len();
  let mut train_data = vec![TrainDatum::origin(); num_of_entries];
  let mut thread_pool = Pool::new(num_of_threads);
  let mut align_feature_score_sets = AlignFeatureCountSets::new(0.);
  align_feature_score_sets.transfer();
  let ref_2_align_feature_score_sets = &align_feature_score_sets;
  thread_pool.scoped(|scope| {
    for (input_file_path, train_datum) in entries.iter().zip(train_data.iter_mut()) {
      scope.execute(move || {
        *train_datum = TrainDatum::<u16>::new(
          &input_file_path.path(),
          min_bpp,
          min_align_prob,
          ref_2_align_feature_score_sets,
        );
      });
    }
  });
  constrain::<u16>(
    &mut thread_pool,
    &mut train_data,
    output_file_path,
    enable_random_init,
    learning_tolerance,
  );
}
