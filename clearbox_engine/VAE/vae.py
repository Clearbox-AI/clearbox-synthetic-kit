import abc


class VAEInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "encode")
            and callable(subclass.encode)
            and hasattr(subclass, "decode")
            and callable(subclass.decode)
            and hasattr(subclass, "save")
        )

    @abc.abstractmethod
    def encode(self):
        """Load in the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self):
        """Load in the data set"""
        raise NotImplementedError
