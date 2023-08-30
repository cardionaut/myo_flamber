"""Extracts data for desired experiment
"""

import os

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig

from excel.analysis.utils.helpers import save_tables
from excel.global_helpers import checked_dir

# from sklearn.experimental import enable_iterative_imputer  # because of bug in sklearn
# from sklearn.impute import IterativeImputer, MissingIndicator


class MergeData:
    """Extracts data for given localities, dims, axes, orientations and metrics"""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.src = config.dataset.out_dir
        dir_name = checked_dir(config.dataset.dims, config.dataset.strict)
        self.checked_src = os.path.join(config.dataset.out_dir, '4_checked', dir_name)
        self.merged_dir = os.path.join(config.dataset.out_dir, '5_merged')
        self.dims = config.dataset.dims
        self.peak_values = config.merge.peak_values
        self.keep_thresh = config.merge.keep_thresh
        self.mdata_src = config.dataset.mdata_src
        self.target_label = config.analysis.experiment.target_label
        self.experiment_name = config.analysis.experiment.name
        self.axes = config.analysis.experiment.axes
        self.orientations = config.analysis.experiment.orientations
        self.metrics = config.analysis.experiment.metrics
        self.metadata = config.analysis.experiment.metadata
        self.segments = config.analysis.experiment.segments

        if self.metadata:  # always want subject IDs and label
            to_add = ['redcap_id', 'pat_id', self.target_label]
            self.metadata.extend([col for col in to_add if col not in self.metadata])
        elif len(self.metadata) == 0:  # special mode to merge all available metadata
            self.metadata = []
        else:
            self.metadata = [self.target_label]

        self.relevant = []
        self.table_name = None

    def __call__(self) -> None:
        logger.info('Merging data according to config parameters...')
        tables_list = []
        self.identify_tables()  # identify relevant tables w.r.t. input parameters
        subjects = os.listdir(self.checked_src)
        for subject in subjects:  # loop over subjects
            self.col_names = []  # OPT: not necessary for each patient
            self.subject_data = pd.Series(dtype='float64')
            for table in self.loop_files(subject):
                if self.peak_values:
                    table = self.remove_time(table)
                    self.extract_peak_values(table)
                else:
                    logger.error('peak_values=False is not implemented yet.')
                    raise NotImplementedError
            tables_list.append(self.subject_data)

        # Build DataFrame from list (each row represents a subject)
        tables = pd.DataFrame(tables_list, index=subjects, columns=self.col_names)
        tables = tables.rename_axis('subject').reset_index()  # add a subject column and reset index

        if self.metadata or len(self.metadata) == 0:  # read and clean metadata
            tables = self.add_metadata(tables)

        tables = tables.sort_values(by='subject')  # save the tables for analysis
        save_tables(out_dir=self.merged_dir, experiment_name=self.experiment_name, tables=tables)

    def __del__(self) -> None:
        logger.info('Data merging finished.')

    def identify_tables(self) -> None:
        for segment in self.segments:
            for dim in self.dims:
                for axis in self.axes:
                    for orientation in self.orientations:
                        if (
                            axis == 'short_axis'
                            and orientation == 'longit'
                            or axis == 'long_axis'
                            and orientation == 'circumf'
                            or axis == 'long_axis'
                            and orientation == 'radial'
                        ):
                            continue  # skip impossible or imprecise combinations

                        for metric in self.metrics:
                            self.relevant.append(f'{segment}_{dim}_{axis}_{orientation}_{metric}')

    def loop_files(self, subject) -> pd.DataFrame:
        for root, _, files in os.walk(os.path.join(self.checked_src, subject)):
            files.sort()  # sort files for consistent order of cols among subjects
            for file in files:
                # consider only relevant tables
                for table_name in self.relevant:
                    if file.endswith('.xlsx') and f'{table_name}_(' in file:
                        # logger.info(f'Relevant table {table_name} found for subject {subject}.')
                        self.table_name = table_name
                        file_path = os.path.join(root, file)
                        table = pd.read_excel(file_path)
                        yield table

    def remove_time(self, table) -> pd.DataFrame:
        """Remove time columns from ROI analysis tables"""
        return table[table.columns.drop(list(table.filter(regex='time')))]

    def extract_peak_values(self, table) -> None:
        """Extract peak values from ROI analysis tables"""
        info_cols = 1 if 'aha' in self.table_name else 2  # AHA data got one info col, ROI data got two info cols
        if 'long_axis' in self.table_name:  # ensure consistent naming between short and long axis
            table = table.rename(columns={'series, slice': 'slice'})

        if 'roi' in self.table_name:  # ROI analysis, remove slice-wise global rows and  keep only global, endo, epi ROI
            table = table.drop(table[(table.roi == 'global') & (table.slice != 'all slices')].index)
            to_keep = ['global', 'endo', 'epi']
            table = table[table.roi.str.contains('|'.join(to_keep)) == True]

        # Circumferential and longitudinal strain and strain rate peak at minimum value
        if 'strain' in self.table_name and ('circumf' in self.table_name or 'longit' in self.table_name):
            peak = table.iloc[:, info_cols:].min(axis=1, skipna=True)  # compute peak values over sample cols
        else:
            peak = table.iloc[:, info_cols:].max(axis=1, skipna=True)

        table = pd.concat([table.iloc[:, :info_cols], peak], axis=1)  # concat peak values to info cols

        if 'roi' in self.table_name:  # ROI analysis -> group by global/endo/epi
            table = table.groupby(by='roi', sort=False).agg('mean', numeric_only=True)  # remove slice-wise global rows

        col_names = []  # store column names for later
        for segment in to_keep:
            orientation = [o for o in self.orientations if o in self.table_name][0]
            metric = [m for m in self.metrics if m in self.table_name][0]
            col_names.append(f'{segment}_{orientation}_{metric}')
            self.col_names.append(f'{segment}_{orientation}_{metric}')

        self.subject_data = pd.concat((self.subject_data, pd.Series(list(table.iloc[:, 0]), index=col_names)), axis=0)

    def add_metadata(self, tables):
        """Add metadata to tables"""
        try:
            mdata = pd.read_excel(self.mdata_src)
        except FileNotFoundError:
            logger.error(f'Metadata file not found, check path: {self.mdata_src}' '\nContinue without metadata...')
            mdata = None

        if mdata is not None:
            if len(self.metadata) != 0:
                mdata = mdata[self.metadata]
            else:
                self.metadata = [mdata for mdata in mdata.columns if not self.target_label.lower() in mdata.lower()]
                self.metadata.append(self.target_label)
                mdata = mdata[self.metadata]
            # clean some errors in metadata
            if 'mace' in self.metadata:  # TODO: add for other mace types as well (e.g. in function)
                mdata.loc[mdata['mace'] == 999, 'mace'] = 0
            if 'MACE_any_chf_hosp' in self.metadata:
                mdata.loc[mdata['MACE_any_chf_hosp'] == 999, 'MACE_any_chf_hosp'] = 0
            if 'lge' in self.metadata:
                mdata.loc[mdata['lge'] == 999, 'lge'] = np.nan
                mdata = mdata.rename(columns={'lge': 'LGE'})
            if 'fhxcad___1' in self.metadata:
                mdata.loc[~mdata['fhxcad___1'].isin([0, 1]), 'fhxcad___1'] = 0
                mdata = mdata.rename(columns={'fhxcad___1': 'T2'})
            mdata = mdata.replace(999, np.nan)
            mdata = mdata.replace('999', '')

            # clean subject IDs
            mdata = mdata[mdata['redcap_id'].notna()]  # remove rows without redcap_id
            mdata['redcap_id'] = mdata['redcap_id'].astype(int).astype(str) + '_rc'
            mdata['pat_id'] = mdata['pat_id'].astype(object).apply(lambda x: str(int(x)) + '_p' if pd.notnull(x) else x)
            mdata['pat_id'].fillna(mdata['redcap_id'], inplace=True)  # patients without pat_id get redcap_id
            mdata = mdata.rename(columns={'pat_id': 'subject'})

            # merge the cvi42 data with available metadata
            tables = tables.merge(mdata, how='inner', on='subject')
            tables = tables.drop('subject', axis=1)  # use redcap_id as subject id
            tables = tables.rename(columns={'redcap_id': 'subject'})

            # remove any metadata columns containing less than self.keep_thresh data
            num_features = len(tables.columns)
            tables = tables.dropna(axis=1, thresh=self.keep_thresh * len(tables.index))
            logger.info(
                f'Removed {num_features - len(tables.columns)} features with less than {int(self.keep_thresh*100)}% data, '
                f'number of remaining features: {len(tables.columns)}'
            )
            assert (
                self.target_label in tables.columns
            ), f'Target label {self.target_label} was removed due to NaN self.keep_thresh.'

            # remove these columns from the metadata list
            self.metadata = [col for col in self.metadata if col in tables.columns]
            self.config.analysis.experiment.metadata = self.metadata

            # remove any subject row containing less than self.keep_thresh data
            num_subjects = len(tables.index)
            tables = tables.dropna(axis=0, thresh=self.keep_thresh * len(tables.columns))
            logger.info(
                f'Removed {num_subjects - len(tables.index)} subjects with less than {int(self.keep_thresh*100)}% data, '
                f'number of remaining subjects: {len(tables.index)}'
            )

            # LGE/T2 columns
            # if 'LGE' in tables.columns and 'T2' in tables.columns:
            #     lge_bool = tables['LGE'].astype(bool)
            #     t2_bool = tables['T2'].astype(bool)
            #     tables['LGE+/T2+'] = (lge_bool & t2_bool).astype(int)
            #     tables['LGE+/T2-'] = (lge_bool & (~t2_bool)).astype(int)
            #     tables['LGE-/T2+'] = ((~lge_bool) & t2_bool).astype(int)
            #     tables['LGE-/T2-'] = ((~lge_bool) & (~t2_bool)).astype(int)

            # Remove features containing the same value for all patients
            # nunique = tables.nunique()
            # cols_to_drop = nunique[nunique == 1].index
            # tables = tables.drop(cols_to_drop, axis=1)

        return tables
