import customtkinter as ctk
from pages.main_page import MainPage
from pages.guide_page import GuidePage
from pages.about_us_page import AboutUs
from components.header_content import Header
from components.theme import Theme

class App(ctk.CTk):
    def __init__(self):
         super().__init__()
         self.title("Hello World")
         self.after(0, lambda: self.state("zoomed"))         
         
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

         #Initialize frames
         self.frames = {}

         for F in (MainPage, GuidePage, AboutUs):
             page_name = F.__name__
             frame = F(parent=container, controller=self)
             self.frames[page_name] = frame
             frame.grid(row=0, column=0, sticky="nsew")
         
         self.show_frame("MainPage")

         #Run this code when clicking X to close app
         self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

        self.update_idletasks()
        self.header.select_button(page_name)
    
    def on_closing(self):
        if "MainPage" in self.frames:
            self.frames["MainPage"].cap.release()
            self.destroy()
    
if __name__ == "__main__":
    app = App()
    app.mainloop()