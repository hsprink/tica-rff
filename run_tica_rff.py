import argparse
from pathlib import Path

from util import load_from_file
from tica_rff import TicaRffModel
from coarse_grained_model import CoarseGrainedModel

parser = argparse.ArgumentParser(description='Run the TICA-RFF pipeline for a given system.')
parser.add_argument('config_name', help="name of the .json file (without extension) to load parameters from, e.g. 'BBA'")
args = parser.parse_args()

config = load_from_file(Path(f'{args.config_name}.json'))

tica_rff_model = TicaRffModel(**config['tica_rff'])
tica_rff_model.run()

coarse_grained_model = CoarseGrainedModel(tica_rff_model, **config['coarse_grained_model'])
coarse_grained_model.run()
