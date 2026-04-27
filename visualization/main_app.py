import customtkinter as ctk
from pages.main_page_zmq import MainPage_zmq
from pages.guide_page import GuidePage
from pages.about_us_page import AboutUs
from components.header_content import Header
from components.theme import Theme

class App(ctk.CTk):
    def __init__(self):
         super().__init__()
         self.title("VIPER")
         self.after(0, lambda: self.state("zoomed"))         
         
         #Header-Content-Footer
         self.grid_rowconfigure(0, weight = 0)
         self.grid_rowconfigure(1, weight = 1)
         self.grid_rowconfigure(2, weight = 0)
         self.grid_columnconfigure(0, weight = 1)

         self.header = Header(self, self)
         self.header.grid(row=0, column=0, sticky="ew")

         container = ctk.CTkFrame(self, fg_color=Theme.TP)
         container.grid(row=1, column=0, sticky="nsew")
         container.grid_columnconfigure(0, weight=1)
         container.grid_rowconfigure(0, weight=1)

         self.footer = ctk.CTkFrame(self, height=110, fg_color=Theme.BLUE, corner_radius=0)
         self.footer.grid(row=2, column=0, sticky="ew")

         #Initialize pages
         self.pages = {}

         for F in (MainPage_zmq, GuidePage, AboutUs):
             page_name = F.__name__
             page = F(parent=container, controller=self)
             self.pages[page_name] = page
             page.grid(row=0, column=0, sticky="nsew")
         
         self.show_page("MainPage_zmq")

    def show_page(self, page_name):
        page = self.pages[page_name]
        page.grid(row=0, column=0, sticky="nsew")
        page.tkraise()

        #Hide other pages than the current one
        for name, f in self.pages.items():
            if name != page_name:
                f.grid_remove()
        
        self.update_idletasks()
        self.header.select_button(page_name)
    
if __name__ == "__main__":
    app = App()
    app.mainloop()