import customtkinter as ctk
from components.theme import Theme

class HeaderButton(ctk.CTkButton):
    def __init__(self, master, text, command):
        super().__init__(
            master,
            text=text,
            command=command,
            width=118,
            height=24,
            fg_color=Theme.WHITE,
            text_color=Theme.BLUE,
            hover_color=Theme.HOVER,
            corner_radius=5
        )

class Header(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=Theme.BLUE, corner_radius=0)
        
        #Stretching of header
        self.grid_columnconfigure(0, weight = 1)
        self.grid_columnconfigure(1, weight = 0)
        self.grid_columnconfigure(2, weight = 1)

        #Title label
        self.label = ctk.CTkLabel(self, text="                                         modelName", font=(Theme.FONT_T,48), text_color=Theme.WHITE)
        self.label.grid(row=0, column=1, padx=(70,10), pady=20)

        #Buttons
        self.btn_group = ctk.CTkFrame(self, fg_color=Theme.TP)
        self.btn_group.grid(row=0, column=2, sticky="e", padx=30)

        self.buttons = {}

        self.buttons["MainPage"] = HeaderButton(self.btn_group,"Main Page",lambda: self.nav_to("MainPage", controller))
        self.buttons["GuidePage"] = HeaderButton(self.btn_group,"Guide",lambda: self.nav_to("GuidePage", controller))
        self.buttons["AboutUs"] = HeaderButton(self.btn_group,"About Us",lambda: self.nav_to("AboutUs", controller))

        for btn in self.buttons.values():
            btn.pack(side="left", expand=True, anchor="w",padx=15)
        
        self.select_button("Main Page")

    def nav_to(self, page_name, controller):
        controller.show_frame(page_name)
        self.select_button(page_name)
    
    def select_button(self, active_page_name):
        for name, btn in self.buttons.items():
            if name == active_page_name:
                btn.configure(fg_color = Theme.HOVER)
            else:
                btn.configure(fg_color = Theme.WHITE)