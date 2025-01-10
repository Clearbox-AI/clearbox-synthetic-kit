import abc


class EngineInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "encode")
            and callable(subclass.encode)
            and hasattr(subclass, "decode")
            and callable(subclass.decode)
            and hasattr(subclass, "save")
            and callable(subclass.save)
            and hasattr(subclass, "fit")
            and callable(subclass.fit)
            and hasattr(subclass, "reconstruction_error")
            and callable(subclass.reconstruction_error)
            and hasattr(subclass, "sample_from_latent_space")
            and callable(subclass.sample_from_latent_space)
        )

    @abc.abstractmethod
    def encode(self):
        """Load in the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self):
        """Load in the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def save(self):
        """Load in the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self):
        """Load in the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def reconstruction_error(self):
        """Load in the data set"""
        raise NotImplementedError

    @abc.abstractmethod
    def sample_from_latent_space(self):
        """Load in the data set"""
        raise NotImplementedError
