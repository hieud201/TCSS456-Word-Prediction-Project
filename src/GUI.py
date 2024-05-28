import tkinter as tk
import VulcanLanguageModel



def main():
    VulcanLanguageModel.init()
    create_window()
    
    
    
def create_window():
    # Create the main window
    root = tk.Tk()
    root.title("Grammarly: Vulcan Edition")

    # Set window size
    root.geometry("500x250")

    # Disable window resizing
    root.resizable(False, False)

    # Create a label widget
    label = tk.Label(root, text="Welcome to Grammarly: Vulcan Edition!", padx=20, pady=10)
    label.pack()

    # Create a textbox widget
    textbox = tk.Text(root, height=10, width=50)
    textbox.pack()
    textbox.bind('<space>', lambda event : on_space(event, textbox))


    # Start the main event loop
    root.mainloop()



def on_text_change(event, textbox):
    if (event.char == ' '):
        return
    
    text = textbox.get(1.0, tk.END).replace("\n", '').split(" ")[-1]
    
    
    # print("Text in the textbox:", text)
    # print(event)
    print(VulcanLanguageModel.predict(text))


def on_space(event, text):
    # Get the current position of the cursor
    cursor_position = text.index(tk.INSERT)

    # Get the text from the start to the cursor position
    text_before_cursor = text.get("1.0", cursor_position)

    # Split the text to get the words and take the last word
    words = text_before_cursor.split()
    if words:
        last_word = words[-1]
        print(f"Last word before space: {last_word}")
        print(VulcanLanguageModel.predict(last_word))

main()