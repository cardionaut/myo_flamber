""" Analysis module for all kinds of experiments
"""

import os
import sys

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from excel.analysis.utils.merge_data import MergeData

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)


class Analysis:
    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.src_dir = config.dataset.out_dir
        self.overwrite = config.merge.overwrite
        self.experiment_name = config.analysis.experiment.name
        self.target_label = config.analysis.experiment.target_label

    def __call__(self) -> None:
        merged_path = os.path.join(self.src_dir, '5_merged', f'{self.experiment_name}.xlsx')

        # Data merging
        if os.path.isfile(merged_path) and not self.overwrite:
            logger.info('Merged data available, skipping merge step...')
        else:
            merger = MergeData(self.config)
            merger()

if __name__ == '__main__':

    @hydra.main(version_base=None, config_path='../../config', config_name='config')
    def main(config: DictConfig) -> None:
        logger.remove()
        logger.add(sys.stderr, level=config.logging_level)
        analysis = Analysis(config)
        analysis()

    main()
