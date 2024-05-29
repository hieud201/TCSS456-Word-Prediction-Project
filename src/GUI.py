import string
import tkinter as tk
from tkinter import font

import VulcanLanguageModel


def main():
    VulcanLanguageModel.init()
    create_window()


def create_window():
    # Create the main window
    root = tk.Tk()
    root.title("Grammarly: Vulcan Edition")

    # Get screen width and height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Calculate window dimensions as one-third of the screen size
    window_width = screen_width // 3
    window_height = screen_height // 3

    # Set the window size
    root.geometry(f"{window_width}x{window_height}")

    # Disable window resizing
    root.resizable(False, False)

    # Create a textbox widget
    text_box = tk.Text(root, height=5, width=50, font=font.Font(size=14))
    text_box.pack(pady=30)
    text_box.bind('<space>', lambda event: on_space(root, text_box, button_frame))

    # Create a frame to hold the buttons
    button_frame = tk.Frame(root)
    # Center-align the frame within the root window
    button_frame.pack(anchor=tk.CENTER)

    # Start the main event loop
    root.mainloop()


def on_space(root, text_box, button_frame):
    # Get the current position of the cursor
    cursor_position = text_box.index(tk.INSERT)

    # Get the text from the start to the cursor position
    text_before_cursor = text_box.get("1.0", cursor_position)

    # Split the text to get the words and take the last word
    words = text_before_cursor.split()
    if words:
        last_word = words[-1]
        if last_word not in string.punctuation:
            three_most_likely_words = VulcanLanguageModel.predict(last_word)
            print(f"Last word before space: {last_word}")
            print(three_most_likely_words)
            # Create three new buttons labeled with three most likely words
            create_new_button(button_frame, text_box, three_most_likely_words)
        else:
            print(f"Last word before space: {last_word}")
            # Create three new buttons labeled with three empty strings
            create_new_button(button_frame, text_box, ('', '', ''))


def create_new_button(frame, text_box, button_texts):
    # new_button = tk.Button(frame, text=button_text, command=lambda: print_to_textbox(text_box, button_text))
    # new_button.pack(side=tk.LEFT, padx=5)

    # Clear the existing buttons in the frame
    for widget in frame.winfo_children():
        widget.destroy()

    # Create and pack new buttons with the provided labels
    for text in button_texts:
        new_button = tk.Button(frame, text=text, command=lambda t=text: print_to_textbox(text_box, t, frame))
        new_button.pack(side=tk.LEFT, padx=5)


def print_to_textbox(text_box, text, button_frame):
    text_box.insert(tk.END, text + " ")
    on_space(None, text_box, button_frame)


# Runs the program
main()
