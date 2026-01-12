from flask import Flask, render_template, request, redirect, url_for
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import pandas as pd
import os

matplotlib.use("Agg")

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
DATA_FILE = os.path.join(UPLOAD_FOLDER, '1.csv')

faculties_data = []
teachers = []
age = {}
income_data = {}
subjects = []
ex_faculties = []
sizes = [5.26] * 16
colors = ['blue', 'red', 'green', 'purple', 'orange', 'pink', 'brown', 'gray']

@app.route('/')
def upload_page():
    return render_template('index1.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        file.save(DATA_FILE)
        read_data()
        return redirect(url_for('index'))

def read_csv():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame()

def read_data():
    global df, teachers, age, income_data, subjects, ex_faculties, faculties_data
    global c_branchs, c_faculties, x_faculties, c_jobRole, c_sub, max_income

    df = read_csv()

    if df.empty:
        return  

    faculties_data.clear()
    teachers.clear()
    age.clear()
    income_data.clear()
    subjects.clear()
    ex_faculties.clear()

    for i in range(len(df)):
        teacher_name = df.iloc[i, 0]
        teachers.append(teacher_name)
        data = {'name': teacher_name}
        
        age[teacher_name] = df.iloc[i, 7]
        subjects.append(df.iloc[i, 2])
        income_data[teacher_name] = df.iloc[i, 4]

        if df.iloc[i, 6] == 'Ex-Faculties':
            ex_faculties.append(teacher_name)

        result = '' if pd.isna(df.iloc[i, 5]) else df.iloc[i, 5]

        if df.iloc[i, 3] == 'Male':
            data.update({'male_head_od': result, 'male_first_job': df.iloc[i, 8], 'female_head_od': '', 'female_first_job': ''})
        else:
            data.update({'male_head_od': '', 'male_first_job': '', 'female_head_od': result, 'female_first_job': df.iloc[i, 8]})

        faculties_data.append(data)

    c_branchs = len(df['Branch'].unique())
    c_faculties = len(teachers)
    x_faculties = len(ex_faculties)
    c_jobRole = len(df['Job role'].unique())
    c_sub = len(subjects)
    max_income = df['Income'].max()

read_data()

def faculty_chart():
    faculties = teachers
    values = [1] * len(faculties)  

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.barh(faculties, values, color=['blue', 'red', 'green', 'purple', 'orange', 'pink'])

    for bar, subject in zip(bars, subjects):
        ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                subject, ha='center', va='center', fontsize=7, color='black')
        
    ax.legend(bars, teachers, title="Faculty Names", bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=7)


    ax.set_xlabel("Teaching")
    ax.set_ylabel("Faculties")

    ax.set_xticks([])

    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    chart_base64 = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)

    return chart_base64


def create_donut_chart():
    names = list(age.keys())
    ages = list(age.values())

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(
        ages, labels=ages, autopct='%1.1f%%', startangle=90, pctdistance=0.85
    )

    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    fig.gca().add_artist(centre_circle)

    plt.legend(wedges, names, title="Faculties", loc="center left", bbox_to_anchor=(1, 0.5))

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return img_base64

def create_histogram():
    sorted_data = sorted(income_data.items(), key=lambda x: x[1], reverse=True)
    faculty_names, income_values = zip(*sorted_data)

    colors = ['darkblue' if name in ex_faculties else 'lightblue' for name in faculty_names]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(faculty_names, income_values, color=colors)

    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Faculties", fontsize=12)
    plt.ylabel("Income (in Rs.)", fontsize=12)

    legend_labels = [
    plt.Rectangle((0, 0), 1, 1, color="darkblue", label="Ex-Faculties"),
    plt.Rectangle((0, 0), 1, 1, color="lightblue", label="Faculties"),
    ]
    ax.legend(handles=legend_labels)
    plt.show()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()

    return img_base64


def create_pie_chart():
    fig, ax = plt.subplots(figsize=(3, 3)) 
    ax.set_title("Count of Subject by Faculties", fontsize=15, pad=25)
    wedges, texts = ax.pie(sizes, colors=colors, startangle=90, textprops={'fontsize': 8})

    for wedge, teacher in zip(wedges, sizes):  
        theta = (wedge.theta2 + wedge.theta1) / 2 
        x = np.cos(np.radians(theta)) * 1.2 
        y = np.sin(np.radians(theta)) * 1.2
        ax.text(x, y, teacher, ha='center', va='center', fontsize=8, color="black")

    ax.legend(wedges, teachers, title="Teachers", loc="center left", bbox_to_anchor=(1.1, 0.5), fontsize=8)

    plt.axis('equal') 


    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    return img_base64

@app.route('/show_info')
def index():
    read_data()
    if not teachers:
        return "No data available. Please upload a valid CSV file."

    return render_template('index.html', 
                           pie_chart=create_pie_chart(), 
                           faculties_data=faculties_data,
                           histogram=create_histogram(), 
                           donut_chart=create_donut_chart(),
                           faculty_chart=faculty_chart(), 
                           c_branchs=c_branchs if 'c_branchs' in globals() else 0, 
                           c_faculties=c_faculties if 'c_faculties' in globals() else 0,
                           x_faculties=x_faculties if 'x_faculties' in globals() else 0, 
                           c_jobRole=c_jobRole if 'c_jobRole' in globals() else 0, 
                           c_sub=c_sub if 'c_sub' in globals() else 0, 
                           max_income=max_income if 'max_income' in globals() else 0)

if __name__ == '__main__':
    app.run(debug=True)
