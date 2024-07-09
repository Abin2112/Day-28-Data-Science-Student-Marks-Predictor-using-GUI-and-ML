import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load the trained model
model = joblib.load("student_mark_predictor.pkl")

# Create the GUI
def predict_marks():
    try:
        study_hours = float(entry.get())
        predicted_marks = model.predict([[study_hours]])[0][0]
        result_label.config(text=f"Predicted Marks: {predicted_marks:.2f}")
        update_plot(study_hours, predicted_marks)
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter a valid number for study hours.")

def update_plot(study_hours, predicted_marks):
    ax.clear()
    ax.scatter(df['study_hours'], df['student_marks'], label='Actual Marks')
    ax.plot(df['study_hours'], model.predict(df[['study_hours']]), color='red', label='Regression Line')
    ax.scatter([study_hours], [predicted_marks], color='green', s=100, label='Predicted Mark')
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Marks")
    ax.set_title("Study Hours vs Marks")
    ax.legend()
    canvas.draw()

# Load data for plotting
path = "C:/Users/abin/Downloads/student_info.csv"  # Update this with the actual path to your CSV file
df = pd.read_csv(path)
df = df.fillna(df.mean())  # Handle missing values

# Initialize the main window
root = tk.Tk()
root.title("Student Marks Predictor")
root.geometry("800x600")

# Load background image
bg_image = Image.open("C:/Users/abin/OneDrive/Desktop/gui/college3.jpg")
bg_image = bg_image.resize((1900, 950), Image.LANCZOS)
bg_image = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(root, image=bg_image)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create a frame for the widgets to center them
frame = tk.Frame(root, bg='beige')
frame.place(relx=0.5, rely=0.5, anchor='center')

# Create and place the widgets
font = ("Helvetica", 14, "bold")

tk.Label(frame, text="Enter Study Hours:", font=font, bg="beige", fg="black").pack(pady=10)
entry = tk.Entry(frame, font=font)
entry.pack(pady=10)

predict_button = tk.Button(frame, text="Predict Marks", command=predict_marks, font=font)
predict_button.pack(pady=10)

result_label = tk.Label(frame, text="", font=font, bg="beige", fg="black")
result_label.pack(pady=10)

# Create Matplotlib figure and axis
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().place(relx=0.5, rely=0.8, anchor='center', width=760, height=300)

# Initial plot
ax.scatter(df['study_hours'], df['student_marks'], label='Actual Marks')
ax.plot(df['study_hours'], model.predict(df[['study_hours']]), color='red', label='Regression Line')
ax.set_xlabel("Study Hours")
ax.set_ylabel("Marks")
ax.set_title("Study Hours vs Marks")
ax.legend()

# Run the GUI event loop
root.mainloop()
