from kivy.uix.button import Button
from kivy.graphics import Color, RoundedRectangle

from ..constant.color import colors


class RoundButton(Button):
    def __init__(self,
                 main_color: tuple[float, ...],
                 color_after_pressed: tuple[float, ...],
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.rounded_button: RoundedRectangle = RoundedRectangle()

        self.background_color: tuple[float, ...] = colors.INVISIBLE
        self.main_color: tuple[float, ...] = main_color
        self.color_after_press: tuple[float, ...] = color_after_pressed
        self.highlighted_color: tuple[float, ...] = self._calculate_highlighted_color()

        self.times_pressed: int = 0

        self.bind(pos=self.update_shape, size=self.update_shape)
        self.bind(on_press=self.on_press)
        self.bind(on_release=self.on_release)

        self.update_shape()

    def _calculate_highlighted_color(self) -> tuple[float, float, float, float]:
        """
        Calculates the highlighted color of the button by increasing the value of each RGB value of the main color.
        """
        return (
            (self.main_color[0] + 0.150),
            (self.main_color[1] + 0.150),
            (self.main_color[2] + 0.150),
            1.0
        )

    def update_shape(self, *_args) -> None:
        """
        Updates the shape of a shape.
        """
        self.canvas.before.clear()

        with self.canvas.before:
            Color(*self.highlighted_color)
            self.rounded_button: RoundedRectangle = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[25]  # maximale rondheid
            )

    def on_press(self, *args) -> None:
        """
        Changes the button color when it's pressed.
        """
        self.times_pressed += 1

        self.update_shape()

    def on_release(self, *args) -> None:
        with self.canvas.before:
            new_button_color: tuple[float | int, ...] = self.color_after_press

            if ((self.times_pressed % 3) == 0):  # lost registratiefout op
                new_button_color = self.main_color

            Color(*new_button_color)

            self.rounded_button: RoundedRectangle = RoundedRectangle(
                pos=self.pos,
                size=self.size,
                radius=[25]
            )
