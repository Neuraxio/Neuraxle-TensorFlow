from abc import ABCMeta, abstractmethod
from typing import Any


class Observable:
    def __init__(self):
        self._observers = set()

    def subscribe(self, observer: 'Observer'):
        self._observers.add(observer)

    def unsubscribe(self, observer: 'Observer'):
        self._observers.discard(observer)

    def on_next(self, value: Any):
        for observer in self._observers:
            observer.on_next(value)


class Observer(metaclass=ABCMeta):
    @abstractmethod
    def on_next(self, arg: Any):
        pass