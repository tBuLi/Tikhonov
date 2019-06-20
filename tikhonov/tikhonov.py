# -*- coding: utf-8 -*-

"""Main module."""
import symfit as sf
import numpy as np

from tikhonov.models import all_models

class Regularizer:
    def __init__(self, *, model=None, **data):
        if model is None:
            model = self.determine_model(data)
        self.model = model

        self.data = data  # Needed for the shape property
        self.data = self.set_default(data)

    def set_default(self, data):
        """
        Apply the default data.
        :param data:
        :return:
        """
        N_x, N_y, N_sets = self.shapes
        for s, value in self.model.optional_symbols.items():
            if s.name not in data:
                if isinstance(value, sf.Identity):
                    if value.shape[0].name == 'N_y':
                        N = N_y
                    elif value.shape[0].name == 'N_x':
                        N = N_x
                    data[s.name] = np.eye(N)
                else:
                    raise NotImplementedError(
                        'Don\'t know how to handle the default value for {}, '
                        'nobody thought me how!'.format(s)
                    )
        return data

    @property
    def shapes(self):
        """
        Return the shapes based on the data.

        :param data:
        :return: ``N_x, N_y, N_sets``
        """
        # First, infer the shapes N_y and optionally N_x
        N_y, N_sets = self.data['y'].shape  # Always present
        N_x = None
        for var_name in self.data:
            if '_x' in var_name:
                N_x = self.data[var_name].shape[0]
        return N_x, N_y, N_sets

    def determine_model(self, data):
        for model in all_models:
            # The in and output variables should be present in the data.
            io_variables = set(y.name for y in model.dependent_vars)
            io_variables.update(set(x.name for x in model.independent_vars))
            data_keys = set(data.keys())
            optional_keys = set(s.name for s in model.optional_symbols)

            for var in optional_keys:
                if var in io_variables and not var in data_keys:
                    io_variables -= {var}

            diff = data_keys ^ io_variables
            if not diff:
                # Perfect match, let's use this one!
                return model
        # We should not be here, no match was found!
        msg = ('No matching model was found for the data provided ({}).'
               'Please check the input or provide a model '
               'explicitly.').format(list(data.keys()))
        raise TypeError(msg)

    def execute(self):
        """
        Hand the rains to :mod:`symfit`, and regularize that inversion.
        """
        *_, N_sets = self.shapes
        fit_results = []
        for i in range(N_sets):
            # Select the right component
            data = self.data.copy()
            data['y'] = data['y'][:, i]
            data['d'] = data['d'][:, i]
            fit = sf.Fit(self.model, **data)
            fit_results.append(fit.execute())
        if N_sets == 1:
            return fit_results[0]
        else:
            return fit_results
