from abc import ABC, abstractmethod

STRATEGY_REGISTRY = {}


def register_strategy(cls):
    STRATEGY_REGISTRY[cls.__name__] = cls
    return cls


class Strategy(ABC):

    @abstractmethod
    def ordering(self, reqs):
        pass


@register_strategy
class FIFOStrategy(Strategy):
    def ordering(self, reqs):
        return sorted(reqs, key=lambda x: x.id)


@register_strategy
class ShortestFirstStrategy(Strategy):
    def ordering(self, reqs):
        return sorted(reqs, key=lambda x: x.length, reverse=True)


@register_strategy
class LongestFirstStrategy(Strategy):
    def ordering(self, reqs):
        return sorted(reqs, key=lambda x: x.length)


@register_strategy
class LIFOStrategy(Strategy):
    def ordering(self, reqs):
        return sorted(reqs, key=lambda x: x.id, reverse=True)


def select_swapped_reqs(reqs, strategy, need_tokens):
    sorted_reqs = strategy.ordering(reqs)

    swapped_reqs = []
    used_tokens = 0
    idx = 0

    for idx, req in enumerate(sorted_reqs):
        used_tokens += req.length
        swapped_reqs.append(req)
        if used_tokens > need_tokens:
            break

    return swapped_reqs
