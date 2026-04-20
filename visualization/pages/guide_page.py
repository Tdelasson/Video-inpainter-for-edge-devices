import customtkinter as ctk
from components.header_content import Header
from components.text import TitleText, BodyText
from components.theme import Theme

class GuidePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=Theme.WHITE)
        
        self.grid_columnconfigure(0, weight=1)

        self.main_content = ctk.CTkScrollableFrame(self, fg_color=Theme.WHITE, label_text="", 
                         orientation="vertical",
                         width=800,
                         height=600)
        self.main_content.grid(row=0, column=0, sticky="nsew")
        self.main_content.grid_columnconfigure(0, weight=1)

        self.title = TitleText(self.main_content, text="Guide")
        self.title.grid(row=0, column=0, padx=40, pady=20, sticky="w")

        description = ("""Setup:
1. 
2. 
                       r
                       r
                       r
                       r
                       r
                       rr
                       rr

                       r
                       r
                       r
                       r
                       r
                       r

                       r
                       r
                       r
                       r
                       r
                       r
                       r
                       r
                       r
                       r
                       r
                       r
                       r
                       r
                       r
                       r
                       r
                       r
                       rr

"""
                    )
 
        self.content = BodyText(self.main_content, text=description)
        self.content.grid(row=1, column=0, padx=40, pady=0, sticky="w")


        # description = ("Guide Page")
        # self.content = BodyText(self, text=description)
        # self.content.grid(row=0, column=0, pady=20, padx=20, sticky="w")

    