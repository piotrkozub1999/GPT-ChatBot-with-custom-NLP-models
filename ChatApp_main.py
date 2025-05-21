import customtkinter as ctk
from tkinter import filedialog
import threading
import time
import torch
from CustomGPT_Training import load_vocab
from CustomGPT import GPT
from CustomGPT_TextGenerator import customGPT_generate_text
from PretrainedGPT_TextGenerator import GPT2TextGenerator, BaseGPT2TextGenerator, FineTunedGPT2TextGenerator
from config import (DEVICE, BLOCK_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, DROPOUT, MAX_LENGTH, TEMPERATURE,
                    REPETITION_PENALTY, SELECTED_MODELS, MESSAGE_FONT, TYPING_BUBBLE_FONT, BUTTON_FONT, LABEL_FONT)


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ChatApp(ctk.CTk):
    def __init__(self, customGPT_model, customGPT_tokenizer, customGPT_text_generator, pretrainedGPT, pretrainedGPT_lora):
        super().__init__()
        self.selected_model = SELECTED_MODELS[0]
        self.custom_model = customGPT_model
        self.custom_tokenizer = customGPT_tokenizer
        self.custom_text_generator = customGPT_text_generator
        self.pretrained_model = pretrainedGPT
        self.pretrained_model_lora = pretrainedGPT_lora

        self.max_length = MAX_LENGTH  # Default max length
        self.temperature = TEMPERATURE  # Default temperature
        self.repetition_penalty = REPETITION_PENALTY
        self.typing_bubble_frame = None
        self.typing_active = False
        # Variable to store PDF file path
        self.pdf_path = None # Will be added after loading file

        self.title("NLP Chat Application")
        self.geometry("1000x700")

        # Chat display frame
        self.chat_frame = ctk.CTkScrollableFrame(self, width=780, height=600)
        self.chat_frame.grid(row=0, column=0, rowspan=20, columnspan=2, padx=10, pady=10)

        # User input field
        self.user_input = ctk.CTkEntry(self, width=630, placeholder_text="Type your message here...", font=MESSAGE_FONT)
        self.user_input.grid(row=21, column=0, rowspan=2, padx=10, pady=10)
        self.user_input.bind("<Return>", self.send_message)

        # Send button
        self.send_button = ctk.CTkButton(self, width=125, text="Send", command=self.send_message, font=BUTTON_FONT)
        self.send_button.grid(row=21, column=1, rowspan=2, padx=10, pady=10)

        # Clear button
        self.clear_button = ctk.CTkButton(self, width=125, text="Clear output", fg_color="#2b404e",  command=self.clear_chat, font=BUTTON_FONT)
        self.clear_button.grid(row=21, column=3, columnspan=3, padx=15, pady=10, sticky="w")

        # Temperature control
        self.temperature_label = ctk.CTkLabel(self, text=f"Temperature: {self.temperature}", font=LABEL_FONT)
        self.temperature_label.grid(row=1, column=3, columnspan=3, padx=15, pady=0)

        self.temperature_entry = ctk.CTkEntry(self, width=125, font=LABEL_FONT)
        self.temperature_entry.insert(0, str(self.temperature))
        self.temperature_entry.grid(row=2, column=3, columnspan=3, padx=15, pady=0, sticky="n")
        self.temperature_entry.bind("<FocusOut>", self.update_temperature)

        # Max length control
        self.max_length_label = ctk.CTkLabel(self, text=f"Max Length: {self.max_length}", font=LABEL_FONT)
        self.max_length_label.grid(row=3, column=3, columnspan=3, padx=15, pady=0)

        self.max_length_entry = ctk.CTkEntry(self, width=125, font=LABEL_FONT)
        self.max_length_entry.insert(0, str(self.max_length))
        self.max_length_entry.grid(row=4, column=3, columnspan=3, padx=15, pady=0, sticky="n")
        self.max_length_entry.bind("<FocusOut>", self.update_max_length)

        # Model selection via radio buttons
        self.model_var = ctk.StringVar(value="CustomGPT")

        self.model_label = ctk.CTkLabel(self, text="Select Model:", font=LABEL_FONT)
        self.model_label.grid(row=8, column=3, columnspan=3, padx=15, pady=5, sticky="s")

        self.custom_gpt_radio = ctk.CTkRadioButton(
            self, text="CustomGPT", variable=self.model_var, value="CustomGPT", command=self.switch_model
        )
        self.custom_gpt_radio.grid(row=9, column=3, columnspan=3, padx=15, pady=0, sticky="w")

        self.gpt2_radio = ctk.CTkRadioButton(
            self, text="GPT2 - Basic", variable=self.model_var, value="GPT2", command=self.switch_model
        )
        self.gpt2_radio.grid(row=10, column=3, columnspan=3, padx=15, pady=0, sticky="w")

        self.gpt2_pdf_radio = ctk.CTkRadioButton(
            self, text="GPT2 - Fine-Tuned", variable=self.model_var, value="GPT2 LORA", command=self.switch_model
        )
        self.gpt2_pdf_radio.grid(row=11, column=3, columnspan=3, padx=15, pady=0, sticky="w")


        # Load PDF Label
        self.gtp2_pdf_label = ctk.CTkLabel(self, height=20, text="Load PDF:", font=LABEL_FONT)
        self.gtp2_pdf_label.grid(row=16, column=3, padx=15, columnspan=2, pady=0, sticky="s")

        self.delete_pdf_button = ctk.CTkButton(self, width=20, height=20, text="X", fg_color="#c51f1f", command=self.delete_pdf,
                                               font=("Arial", 14, "bold"))
        self.delete_pdf_button.grid(row=16, column=5, padx=0, pady=0, sticky="s")

        # Entry field to display the selected file
        self.pdf_entry = ctk.CTkEntry(self, width=125, height=50,placeholder_text="Browse file",
                                      fg_color='#343638', text_color="black", font=LABEL_FONT, state="readonly")
        self.pdf_entry.grid(row=17, column=3, columnspan=3, padx=15, pady=0, sticky="w")

        # Browse button
        self.browse_pdf_button = ctk.CTkButton(self, width=60, height=25, text="Browse", command=self.browse_pdf, font=BUTTON_FONT)
        self.browse_pdf_button.grid(row=18, column=4, columnspan=2, padx=0, pady=0, sticky="n")


    def send_message(self, event=None):
        user_message = self.user_input.get().strip()
        if not user_message:
            return
        # elif self.selected_model == SELECTED_MODELS[2] and self.pdf_path == None:
        #     self._add_message_bubble("Upload PDF file first", sender="System")
        #     return
        # User massage display
        self.add_message_bubble(user_message, sender="You")
        self.user_input.delete(0, ctk.END)

        # Showing typing bubble while waiting for response
        self.typing_active = True
        self.typing_bubble_frame = self.add_message_bubble(" . . . ", sender="Model")

        # Starting generating response in other thread
        threading.Thread(target=self.generate_model_response, args=(user_message,), daemon=True).start()

        # Checking if typing bubble was deleted
        self.check_typing_bubble()


    def clear_chat(self):
        #Clear all massages from chat window
        for widget in self.chat_frame.winfo_children():
            widget.destroy()


    def check_typing_bubble(self):
        """Check if typing bubble should be removed."""
        if not self.typing_active and self.typing_bubble_frame:
            self.remove_typing_bubble()
        else:
            self.chat_frame.after(100, self.check_typing_bubble)


    def generate_model_response(self, user_message):
        model_response = "Something goes wrong with model selection"
        try:
            # short delay if massage is generated immediately
            time.sleep(0.5)
            if self.selected_model == SELECTED_MODELS[0]:
                model_response = self.custom_text_generator(
                    self.custom_model, self.custom_tokenizer, device=DEVICE,
                    prompt=user_message, max_length=self.max_length,
                    temperature=self.temperature, repetition_penalty=self.repetition_penalty
                )
            elif self.selected_model == SELECTED_MODELS[1]:
                if self.pdf_path is not None:
                    model_response = self.pretrained_model.generate_text_from_pdf(user_message, self.max_length,
                                                                                  self.temperature)
                else:
                    model_response = self.pretrained_model.generate_text(
                        prompt=user_message, max_length=self.max_length,
                        temperature=self.temperature, repetition_penalty=self.repetition_penalty
                    )
            elif self.selected_model == SELECTED_MODELS[2]:
                if self.pdf_path is not None:
                    model_response = self.pretrained_model_lora.generate_text_from_pdf(user_message, self.max_length,
                                                                                       self.temperature)
                else:
                    model_response = self.pretrained_model_lora.generate_text(
                        prompt=user_message, max_length=self.max_length,
                        temperature=self.temperature, repetition_penalty=self.repetition_penalty
                    )
        finally:
            self.typing_active = False
            self.chat_frame.after(0, self.show_model_response, model_response)


    def show_model_response(self, model_response):
        """Show model response after typing bubble."""
        self.remove_typing_bubble()
        # 250ms of delay before showing response for better visualization
        self.chat_frame.after(250, lambda: self.add_message_bubble(model_response, sender="Model"))
        # self._add_message_bubble(model_response, sender="Model")
        self.chat_frame.after(1, self.chat_frame._parent_canvas.yview_moveto, 1)


    def remove_typing_bubble(self):
        """Remove the typing bubble frame without GUI glitches."""
        if self.typing_bubble_frame:
            self.typing_bubble_frame.pack_forget()
            self.typing_bubble_frame.destroy()
            self.typing_bubble_frame = None


    def add_message_bubble(self, message: str, sender: str = "Model"):
        bubble_frame = ctk.CTkFrame(
            self.chat_frame, corner_radius=15,
            fg_color="#1E88E5" if sender == "You" else "#455A64"
        )
        bubble_frame.pack(
            pady=5, padx=20 if sender == "You" else 5,
            anchor="e" if sender == "You" else "w"
        )

        # Adjust bubble width dynamically based on text length
        text_width = min(max(len(message) * 7, 500), 500)  # Adjust width based on length

        message_label = ctk.CTkLabel(
            bubble_frame, text=message, wraplength=text_width, justify="left", text_color="white",
            font=TYPING_BUBBLE_FONT if self.typing_active else MESSAGE_FONT
        )
        message_label.pack(padx=10, pady=5)
        self.chat_frame.after(1, self.chat_frame._parent_canvas.yview_moveto, 1)

        return bubble_frame


    def update_temperature(self, event=None):
        try:
            new_temperature = float(self.temperature_entry.get().strip())
            if 0.0 <= new_temperature <= 1.0:
                self.temperature = new_temperature
                self.temperature_label.configure(text=f"Temperature: {self.temperature}")
            else:
                self.add_message_bubble("Error: Temperature must be between 0.0 and 1.0", sender="System")
        except ValueError:
            self.add_message_bubble("Error: Invalid temperature value", sender="System")


    def update_max_length(self, event=None):
        try:
            new_max_length = int(self.max_length_entry.get().strip())
            if new_max_length > 0:
                self.max_length = new_max_length
                self.max_length_label.configure(text=f"Max Length: {self.max_length}")
            else:
                self.add_message_bubble("Error: Max length must be a positive integer", sender="System")
        except ValueError:
            self.add_message_bubble("Error: Invalid max length value", sender="System")


    def switch_model(self):
        selected_model = self.model_var.get()
        if selected_model == "GPT2":
            # Load Hugging Face GPT2 model
            self.selected_model = SELECTED_MODELS[1]
            self.add_message_bubble("Switched to pretrained GPT-2 Large model.\n"
                                     "You can upload pdf file and ask questions about it.", sender="System")
        elif selected_model == "GPT2 LORA":
            self.selected_model = SELECTED_MODELS[2]
            self.add_message_bubble("Switched to pretrained GPT-2 Large model fine-tuned on Squad dataset.\n"
                                     "You can upload pdf file and ask questions about it.", sender="System")
        else:
            # Load CustomGPT model
            self.selected_model = SELECTED_MODELS[0]
            self.add_message_bubble("Switched to CustomGPT model.", sender="System")


    def browse_pdf(self):
        """Open file explorer to select a PDF file."""
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.set_pdf_path(file_path)


    def set_pdf_path(self, file_path):
        self.pdf_path = file_path
        self.pdf_entry.configure(state="normal")
        self.pdf_entry.delete(0, ctk.END)
        self.pdf_entry.insert(0, file_path[file_path.rindex('/')+1:])
        self.pdf_entry.configure(state="readonly", fg_color="#A8D5BA")
        GPT2TextGenerator.load_pdf(self.pdf_path)
        self.add_message_bubble(f"Loaded PDF: {file_path}", sender="System")


    def delete_pdf(self):
        if self.pdf_path is not None:
            GPT2TextGenerator.clear_pdf()
            self.pdf_path = None
            self.pdf_entry.configure(state="normal", fg_color='#343638')
            self.pdf_entry.delete(0, ctk.END)
            self.pdf_entry.configure(state="readonly")
            self.add_message_bubble("PDF file has been deleted", sender="System")


if __name__ == "__main__":

    # Initializing custom model
    customGPT_tokenizer = load_vocab("./tokenizer/Tokenizer_WikiText103_60k.json")
    VOCAB_SIZE = customGPT_tokenizer.get_vocab_size()
    MODEL_SAVE_PATH = "./Models/gpt_model_WikiText103_Squad_25Epochs.pth"

    customGPT_model = GPT(VOCAB_SIZE, EMBEDDING_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE, DROPOUT).to(DEVICE)
    customGPT_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    customGPT_model.eval()

    # Initializing pretrained GPT model
    model_name = "gpt2-large"
    pretrainedGPT_generator = BaseGPT2TextGenerator(model_name)
    pretrainedGPT_generator_lora = FineTunedGPT2TextGenerator(model_name)

    app = ChatApp(customGPT_model, customGPT_tokenizer, customGPT_generate_text, pretrainedGPT_generator, pretrainedGPT_generator_lora)
    app.mainloop()
