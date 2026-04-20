from tkinter import Canvas

import customtkinter as ctk
from components.header_content import Header
from components.text import TitleText, BodyText
from components.theme import Theme

class AboutUs(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=Theme.WHITE)
        self.grid_columnconfigure(0, weight=1)

        self.title = TitleText(self, text="About Us")
        self.title.grid(row=0, column=0, padx=40, pady=20, sticky="w")

        description = ("""This product is developed by a team of 6th-semester Computer Science students at Aalborg University as part of their Bachelor’s project in Spring 2026.
                       
The tool is designed to perform video inpainting on an edge device, specifically and Nvidia Jetson platform, which remains a relatively unexplored area.

This product contributes to the following points:
*
*
*

The developers are:
Astrid Helene Bak, Dandan Zhao, Jacob Kramer Kaae, Tobias Rosenkrantz de Lasson."""
                    )
 
        self.content = BodyText(self, text=description)
        self.content.grid(row=1, column=0, padx=40, pady=0, sticky="w")


    