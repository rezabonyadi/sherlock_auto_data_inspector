from abc import ABC, abstractmethod
from sherlock.sherlocktabular import SherlockTabularDataModel


class SherlockTabularBase(ABC):
    '''
    This class is the base for all Sherlock functionalities. Each functionality of Sherlock has a computation
    part, done in the method :explore, and visualization done in :visualize method. Each instance of the Sherlock
    functionalities implements these two methods.
    '''
    settings = {}

    def __init__(self, data: SherlockTabularDataModel, settings, logger):
        self.logger = logger
        self.data = data
        if settings is not None:
            for k in settings.keys():
                self.settings[k] = settings[k]
        self.messages = []
        super().__init__()

    @abstractmethod
    def explore(self):
        '''
        This method performs computations related to this Sherlock functionality, and return the results if necessary.
        :return: Results of the computation, if needed.
        '''
        pass

    @abstractmethod
    def visualize(self, show_report=True):
        '''
        This method provides visualization objects, uses what was calculated in the explore method.
        :param show_report: Show the report or just return it.
        :return a set of visualization reports, if neccessary.
        '''
        pass

