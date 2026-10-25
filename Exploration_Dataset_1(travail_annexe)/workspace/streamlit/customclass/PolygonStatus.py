from enum import Enum

class PolygonStatus(Enum):
    VALIDE = ("valid", "blue","green")
    INVALIDE = ("invalid", "purple", "purple")
    MANQUANT = ("missing", "red", "orange")

    @property
    def label(self):
        return self.value[0]

    @property
    def color_gt(self):
        return self.value[1]

    @property
    def color_ocr(self):
        return self.value[2]