import tkinter as tk
import VulcanLanguageModel



def main():
    VulcanLanguageModel.init()
    create_window()
    
    
    
def create_window():
    # Create the main window
    root = tk.Tk()
    root.title("Simple GUI")

    # Set window size
    root.geometry("500x250")

    # Disable window resizing
    root.resizable(False, False)

    # Create a label widget
    label = tk.Label(root, text="Hello, GUI!", padx=20, pady=10)
    label.pack()

    # Create a textbox widget
    textbox = tk.Text(root, height=5, width=30)
    textbox.pack()
    textbox.bind('<KeyRelease>', lambda event : on_text_change(event, textbox))


    # Start the main event loop
    root.mainloop()



def on_text_change(event, textbox):
    text = textbox.get(1.0, tk.END).replace("\n", '')
    # print("Text in the textbox:", text)
    # print(event)
    print(VulcanLanguageModel.predict(text))
    
    
    




main()