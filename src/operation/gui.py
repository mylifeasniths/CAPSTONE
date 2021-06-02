from tkinter import * 
import tkinter as tk
from PIL import Image, ImageGrab, ImageOps
import numpy as np
import joblib
import os

MODELS_DIR = 'models'
DIGIT_MODAL = 'digits_svm_model.gz'
OPERATOR_MODAL = 'operators_svm_model.gz'
operators = {
    'add': '+',
    'sub': '-',
    'mul': '*',
    'div': '/',
}

digit_classifier = joblib.load(os.path.join(MODELS_DIR,DIGIT_MODAL))
operator_classifier = joblib.load(os.path.join(MODELS_DIR,OPERATOR_MODAL))

class GUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.title("SVM")
        self.canvas_width = 200
        self.canvas_height = 200
        self.bg_color = "white"
        self.paint_color = "black"
        self.radius = 8
        self.init_canvas()
        self.count = 0
        self.characters = []
        
        
    def init_canvas(self):
        self.canvas = Canvas(self, width=self.canvas_width, height=self.canvas_width, 
            bg = self.bg_color, cursor="cross",borderwidth=3,relief="ridge")
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)
        self.button_predict = tk.Button(self, text = "Predict", command = self.predict)
        self.button_reset = tk.Button(self, text = "Reset", command = self.reset)
        self.label_digit = tk.Label(self, text="", font=("Helvetica"))
        self.label_info = tk.Label(self, text="Draw an digit", font=("Helvetica"))
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.button_predict.grid(row=2, column=0, pady=2)
        self.button_clear.grid(row=2, column=1, pady=2)
        self.button_reset.grid(row=2, column=2, pady=2)
        self.label_digit.grid(row=0, column=1, padx=2, pady=2)
        self.label_info.grid(row=1, column=0, padx=2, pady=2)
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - self.radius), (event.y - self.radius)
        x2, y2 = (event.x + self.radius), (event.y + self.radius)
        self.canvas.create_oval(x1, y1, x2, y2, fill=self.paint_color, outline=self.paint_color)

    def clear_all(self):
        self.canvas.delete("all")
        self.label_digit.configure(text='')
    
    def reset(self):
        self.count = 0
        self.characters = []
        self.label_info.configure(text='Draw a digit')
        self.clear_all()

    def preprocess(self,is_digit):
        fileName = 'input'
        
        x = self.canvas.winfo_x() + self.winfo_x() + 10
        y = self.canvas.winfo_y() + self.winfo_y() + 30
        x1 = x + self.canvas_width
        y1 = y + self.canvas_height
        
        # Screenshot and Save Image in Canvas
        img = ImageGrab.grab().crop((x,y,x1,y1))
        # img.save(fileName + ".jpg")
        
        # Invert Image
        img = ImageOps.invert(img)
        # img.save(fileName + "_negative" + ".jpg")
        
        # Resize and convert it into greyscale
        img = img.resize((28,28))
        img = img.convert('L')
        # img.save(fileName + "_28x28" + ".jpg")
        
        # Convert it into Numpy array
        img = np.array(img)
        img = img.reshape(28*28)
        if(not is_digit):
            img_norm = img.astype('float32')
            img_norm = img_norm / 255.0
            return img_norm
        return img
    
    def predict(self):
        is_digit = (self.count % 2 == 0)
        img = self.preprocess(is_digit)
        output = ''
        if(is_digit):
            result = digit_classifier.predict([img])
            text = 'Now draw an operator'
            output = f'{result[0]}'
            self.characters.append(output)
        else:
            result = operator_classifier.predict([img])
            text = 'Now draw an digit'
            output = f'{result[0]}'
            self.characters.append(output)

        self.label_digit.configure(text=output)
        self.label_info.configure(text=text)

        if(self.count < 2):
            self.count += 1
        else:
            characters = self.characters
            operation_text = f'{characters[0]} {operators[characters[1]]} {characters[2]}'
            operation_output = calculate(characters[0],characters[2],characters[1])
            text = f'{operation_text} = {operation_output}'
            self.label_info.configure(text=text)
        
def calculate(digit_1,digi_2,operator):
    if(operator == 'add'):
        return int(digit_1) + int(digi_2)
    if(operator == 'sub'):
        return int(digit_1) - int(digi_2)
    if(operator == 'mul'):
        return int(digit_1) * int(digi_2)
    if(operator == 'div'):
        return int(digit_1) / int(digi_2)

def main():
    app = GUI()
    mainloop()

if __name__ == "__main__":
    main()