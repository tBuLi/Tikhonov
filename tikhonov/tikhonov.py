# -*- coding: utf-8 -*-

"""Main module."""
import symfit as sf
import numpy as np
from symfit.core.support import key2str
from sympy.core.numbers import Zero, One

from tikhonov.models import all_models
from tikhonov.definitions import y, d, k, N_x, N_y, N_sets

class Regularizer(sf.Fit):
    def __init__(self, *, model=None, data, **fit_options):
        if model is None:
            model = self.determine_model(data)
        self.shapes = self.compute_shapes(data)
        data = self.set_default(model, data)
        self.fit_options = fit_options
        super(Regularizer, self).__init__(model, **key2str(data), **fit_options)

    def set_default(self, model, data):
        """
        Apply the default data.
        :param data:
        :return:
        """
        for s, value in model.optional_symbols.items():
            if s not in data:
                # Build the shape for the numpy arrays. Get it's value from
                # self.shape if it is one of the known symbols. If not, just
                # use the value of N itself since it is already an int in that
                # case.
                num_shape = [self.shapes.get(N, N) for N in s.shape]
                if value is Zero:
                    data[s] = np.zeros(num_shape)
                elif value is One:
                    data[s] = np.ones(num_shape)
                elif isinstance(value, sf.Identity):
                    data[s] = np.eye(num_shape[0])
                else:
                    raise NotImplementedError(
                        'Don\'t know how to handle the default value for {}, '
                        'nobody thought me how!'.format(s)
                    )
        return data

    @staticmethod
    def compute_shapes(data):
        """
        Return the shapes based on the data.

        :param data:
        :return: dict with ``N_x, N_y, N_sets`` as keys.
        """
        # First, infer the shapes N_y and optionally N_x
        _N_y, _N_sets = data[y].shape  # Always present
        _N_x = None
        for var in data:
            if '_x' in var.name:
                _N_x = data[var].shape[0]
        return {N_x: _N_x, N_y: _N_y, N_sets: _N_sets}

    def determine_model(self, data):
        """
        Based on the data, find the model which is best suited to this
        situation.

        :param data:
        :return: a :class:`~symfit.core.models.BaseModel` instance.
        """
        for model in all_models:
            # The in and output variables should be present in the data.
            io_variables = set(model.dependent_vars)
            io_variables.update(set(model.independent_vars))
            data_keys = set(data.keys())
            optional_keys = set(model.optional_symbols)

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

    def generator_execute(self, *, k_start=0, **minimizer_kwargs):
        """
        Hand over the rains to :mod:`symfit`, and regularize that inversion. Use
        this in case of a very expensive calculation, so the results can e.g.
        be written to disk.

        :param k_start:
        :param minimizer_kwargs:
        :return: generator over the subfits per data set, starting from dataset
            ``k_start``.
        """
        # Select the symbols which depend on all the data sets
        global_symbols = []
        local_symbols = {}
        for s in self.model.ordered_symbols:
            if isinstance(s, sf.MatrixSymbol) and N_sets in s.shape:
                global_symbols.append(s)
                # Turn into a vector
                local_symbols[s] = s.func(*s.args[:-1], 1)

        local_model = self.localize_model(local_symbols)

        for set_index in range(k_start, self.shapes[N_sets]):
            # Select the right component
            data = self.data.copy()
            for s in global_symbols:
                if s in data:
                    # Select the component, but maintain shape
                    data[s] = data[s][:, set_index][..., None]

            fit = sf.Fit(local_model, **key2str(data), **self.fit_options)
            fit_result = fit.execute(**minimizer_kwargs)

            model_ans = fit.model(**key2str(fit.independent_data),
                                  **fit_result.params)
            fit_result.model_ans = model_ans

            yield fit_result

    def execute(self, *, k_start=0, **minimizer_kwargs):
        """
        Execute the regularization. Returns list of the
         :class:`~symfit.core.fit_results.FitResults` as returned by ``symfit``
         for all the different datasets. If this does not make sence for memory
         reasons, consider using ``Regularizer.generator_execute`` instead.

        :param k_start: Optional, start from dataset ``k``. Good when execution
            was interupted.
        :param minimizer_kwargs: all this is passed on to ``symfit``'s
            ``Fit.execute``.
        :return: list of :class:`~symfit.core.fit_results.FitResults`.
        """
        fit_results = list(self.generator_execute(k_start=k_start, **minimizer_kwargs))

        if self.shapes[N_sets] == 1:
            return fit_results[0]
        else:
            return fit_results

    def localize_model(self, local_symbols):
        """
        :return:  Return a version of self.model which has ``N_sets`` set to 1,
            i.e. which can be applied on a per data set level.
        """
        local_model_dict = {}
        local_connectivity_mapping = {}
        for var, expr in self.model.items():
            s = local_symbols.get(var, var)
            try:
                local_model_dict[s] = expr.subs(local_symbols)
            except AttributeError:
                # This happens when the component is non-analytic. This means
                # we assume we are dealing with a CallableNumericalModel
                local_model_dict[s] = expr
                dependencies = self.model.connectivity_mapping[var]
                # Update with the local versions when applicable.
                local_dep = set(local_symbols.get(d, d) for d in dependencies)
                local_connectivity_mapping[s] = local_dep

        if local_connectivity_mapping:
            return self.model.__class__(
                local_model_dict,
                connectivity_mapping=local_connectivity_mapping
            )
        else:
            return self.model.__class__(local_model_dict)
