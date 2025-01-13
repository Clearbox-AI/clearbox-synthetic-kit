import abc


class DiffusionInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "sample")
            and callable(subclass.sample)
            and hasattr(subclass, "save")
        )

    @abc.abstractmethod
    def sample(self):
        """Load in the data set"""
        raise NotImplementedError

