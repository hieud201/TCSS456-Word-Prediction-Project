import tkinter as tk

def on_button_click():
    label.config(text="Button clicked!")
    text = textbox.get(1.0, tk.END)
    print("Text in the textbox:", text)

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

# Create a button widget
button = tk.Button(root, text="Click me!", command=on_button_click)
button.pack()

# Start the main event loop
root.mainloop()
