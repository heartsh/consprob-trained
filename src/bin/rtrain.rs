extern crate consprob;

use consprob::*;
use std::env;

fn main() {
  let args = env::args().collect::<Args>();
  let program_name = args[0].clone();
  let mut opts = Options::new();
  opts.reqopt(
    "i",
    "input_dir",
    "A path to an input directory containing RNA structural alignments in STOCKHOLM format",
    "STR",
  );
  opts.reqopt(
    "o",
    "output_file_path",
    "A path to an output file to record intermediate training log likelihoods",
    "STR",
  );
  opts.optopt(
    "",
    "min_base_pair_prob",
    &format!(
      "A minimum base-pairing-probability (Uses {} by default)",
      DEFAULT_MIN_BPP_4_TRAIN
    ),
    "FLOAT",
  );
  opts.optopt(
    "",
    "offset_4_max_gap_num",
    &format!(
      "An offset for maximum numbers of gaps (Uses {} by default)",
      DEFAULT_OFFSET_4_MAX_GAP_NUM_TRAIN
    ),
    "UINT",
  );
  /* opts.optopt(
    "",
    "num_of_epochs",
    &format!(
      "The number of epochs for training (Uses {} by default)",
      DEFAULT_NUM_OF_EPOCHS,
    ),
    "UINT",
  ); */
  opts.optflag("d", "disables_transfer_learn", "Disables transfer learning to use pre-trained CONTRAfold parameters");
  opts.optopt("t", "num_of_threads", "The number of threads in multithreading (Uses the number of the threads of this computer by default)", "UINT");
  opts.optflag("h", "help", "Print a help menu");
  let matches = match opts.parse(&args[1..]) {
    Ok(opt) => opt,
    Err(failure) => {
      print_program_usage(&program_name, &opts);
      panic!(failure.to_string())
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
    DEFAULT_MIN_BPP_4_TRAIN
  };
  let offset_4_max_gap_num = if matches.opt_present("offset_4_max_gap_num") {
    matches
      .opt_str("offset_4_max_gap_num")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_OFFSET_4_MAX_GAP_NUM_TRAIN
  } as u16;
  /* let num_of_epochs = if matches.opt_present("num_of_epochs") {
    matches
      .opt_str("num_of_epochs")
      .unwrap()
      .parse()
      .unwrap()
  } else {
    DEFAULT_NUM_OF_EPOCHS
  }; */
  let disables_transfer_learn = matches.opt_present("d");
  let num_of_threads = if matches.opt_present("t") {
    matches.opt_str("t").unwrap().parse().unwrap()
  } else {
    num_cpus::get() as NumOfThreads
  };
  let output_file_path = matches.opt_str("o").unwrap();
  let output_file_path = Path::new(&output_file_path);
  let entries: Vec<DirEntry> = read_dir(input_dir_path).unwrap().map(|x| x.unwrap()).collect();
  let num_of_entries = entries.len();
  let mut train_data = vec![TrainDatum::origin(); num_of_entries];
  let mut thread_pool = Pool::new(num_of_threads);
  thread_pool.scoped(|scope| {
    for (input_file_path, train_datum) in entries.iter().zip(train_data.iter_mut()) {
      scope.execute(move || {
        *train_datum = TrainDatum::<u16>::new(&input_file_path.path(), min_bpp);
      });
    }
  });
  rtrain::<u16>(&mut thread_pool, &mut train_data, offset_4_max_gap_num, /* num_of_epochs, */ output_file_path, disables_transfer_learn);
}
