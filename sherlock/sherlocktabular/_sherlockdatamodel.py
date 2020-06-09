import numpy as np
import pandas as pd
from sherlock.sherlockengines import sherlockcomputationhelper as sch


class SherlockTabularDataModel:
    def __init__(self, x: np.array, y: np.array, names=None, turn_to_categorize=False):
        '''

        :param x: Numpy array, rows are observations and columns are dimensions
        :param y: Numpy array, rows are values
        :param names: name of variables. If None, populated by v1 to vn
        '''

        if names is None:
            names = []
            for i in range(x.shape[1]):
                names.append(''.join(['v', str(i)]))
            names = np.asarray(names)

        x, y, names, feasibility = self.check_variables(x, y, names)

        if not feasibility:
            return

        self.is_classification = True
        self.number_variables = x.shape[1]
        self.x = x
        self.y = sch.categorize_response(y) if turn_to_categorize else y
        self.names = names

        self.data_frame = pd.DataFrame(x, columns=names)

        type = []
        for i in range(self.number_variables):
            if sch.is_categorical(self.x[:, i]):
                type.append('c')
            else:
                type.append('n')

        self.x_type = np.asarray(type)

        if sch.is_categorical(self.y):
            self.y_type = 'c'
        else:
            self.y_type = 'n'
            self.is_classification = False

    @staticmethod
    def check_variables(x: np.array, y: np.array, names):
        feasibility = True
        x_new = x.copy()
        y_new = y.copy()
        names_new = names.copy()
        to_remove = []
        for i in range(x.shape[1]):
            if (not sch.is_numeric(x[:, i])) or (sch.is_categorical(x[:, i])):
                print('Sherlock does not support non-numeric/categorical variables as of yet. Excluding ', names[i],
                      ' from variables.')
                to_remove.append(i)
        x_new = np.delete(x_new, to_remove, axis=1)
        names_new = np.delete(names_new, to_remove, axis=0)

        # if not sch.is_categorical(y):
        #     print('Sherlock does not support continuous response as of yet. Digitizing to two levels.')
        #     y_new = sch.categorise_variable(y_new, 2)

        if sch.is_categorical(y) and (len(np.unique(y_new)) > 2):
            print('Sherlock does not support multi-class response as of yet. Digitizing to two levels.')
            y_new = sch.categorise_variable(y_new, 2)

        return x_new, y_new, names_new, feasibility

    def get_variables(self):
        return self.x.copy()

    def get_response(self):
        return self.y.copy()

    def get_names(self):
        return self.names.copy()

    def get_x_types(self):
        return self.x_type

    def get_y_types(self):
        return self.y_type
