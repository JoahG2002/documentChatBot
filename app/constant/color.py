from typing import Final


class Color:
    __slots__: tuple[str, ...] = (
        "MAIN_THEME", "WHITE", "SECONDARY_COLOR", "INVISIBLE", "DARK_GREY", "ERROR_RED", "SECONDARY_COLOR_TRANSPARENT"
    )

    def __init__(self) -> None:
        self.MAIN_THEME: Final[tuple[float, ...]] = (0.2235294117647059, 0.2235294117647059, 0.2235294117647059, 1.0)
        self.SECONDARY_COLOR: Final[tuple[float, ...]] = (0.9411764705882353, 0.611764705882353, 0.3607843137254902, 1.0)
        self.SECONDARY_COLOR_TRANSPARENT: Final[tuple[float, ...]] = (0.9411764705882353, 0.611764705882353, 0.3607843137254902, 0.21)
        self.INVISIBLE: Final[tuple[float, ...]] = (0.0, 0.0, 0.0, 0.0)
        self.WHITE: Final[tuple[float, ...]] = (1.0, 1.0, 1.0, 1.0)
        self.DARK_GREY: Final[tuple[float, ...]] = (0.10588235, 0.10588235, 0.10588235, 1.0)  # 1b1b1b
        self.ERROR_RED: Final[tuple[float, ...]] = (0.8666666666666667, 0.5843137254901961, 0.5725490196078431, 1.0)


colors: Color = Color()
