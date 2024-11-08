from ai_hexagon.model import Model


class MultiDomainCNN(Model):
    """Multi-Domain CNN model performs convolutions in both time and frequency domains."""

    __title__ = "Multi-Domain CNN"
    __authors__ = ["Jiří Klimeš"]

    dims: int = 64
