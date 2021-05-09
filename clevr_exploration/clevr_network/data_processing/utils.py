import pickle, sys

def save_pickle(obj, file):
	with open(file, 'wb') as f:
		pickle.dump(obj, f)

def load_pickle(file):
	with open(file, 'rb') as f:
		obj = pickle.load(f)
	return obj

def extract_args(input_argv=None):
  """
  Pull out command-line arguments after "--". Blender ignores command-line flags
  after --, so this lets us forward command line arguments from the blender
  invocation to our own script.
  """
  if input_argv is None:
    input_argv = sys.argv
  output_argv = []
  if '--' in input_argv:
    idx = input_argv.index('--')
    output_argv = input_argv[(idx + 1):]
  return output_argv


def parse_args(parser, argv=None):
  return parser.parse_args(extract_args(argv))