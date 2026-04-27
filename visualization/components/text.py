import customtkinter as ctk
from components.theme import Theme


class TitleText(ctk.CTkFrame):
    def __init__(self, master, text,):
        super().__init__(master, fg_color=Theme.TP)

        self.label = ctk.CTkLabel(self, text=text, font=(Theme.FONT_T, Theme.FONT_S1), text_color=Theme.BLUE, anchor="w")
        self.label.grid(row=0,column=0, sticky="ew")

        self.line=ctk.CTkFrame(self, height=2, fg_color=Theme.BLUE)
        self.line.grid(row=1,column=0,sticky="ew")

class BodyText(ctk.CTkLabel):
    def __init__(self, master, text, wraplength=1350):
        super().__init__(
            master,
            text=text, 
            font=(Theme.FONT_T,Theme.FONT_S2),
            text_color=Theme.BLUE,
            justify="left",
            wraplength=wraplength,
            anchor="w"
        )

class VideoText(ctk.CTkLabel):
    def __init__(self, master, text):
        super().__init__(
            master,
            text=text,
            font=(Theme.FONT_T, Theme.FONT_S2)
        )
