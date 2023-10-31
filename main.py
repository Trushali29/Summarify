import customtkinter
from customtkinter import StringVar,IntVar
from PIL import Image,ImageTk
from tkinter import filedialog
import tkinter 
from CTkMessagebox import CTkMessagebox
from nltk import sent_tokenize
import Method_1_TF_IDF as tf_idf
import Method_2_Glove as glove
import Method_3_TextRank as textrank
import pre_processing as pp
import model_evaluation as check_acc
import os

title = ''
article = ''
summary = ''
category = ''
article_filename = ''
res_summry = ''
customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"

master = customtkinter.CTk() # customtkinter
master.geometry("1440x780")
master.title("Summarify")
master.resizable(False,False)

# Input File frame
input_text = customtkinter.CTkTextbox(master,font=('arial',15),width = 650 ,height = 400)
input_text.grid(row=7,column=7,columnspan=3,padx=48,pady=30)


# Input File frame
output_summary = customtkinter.CTkTextbox(master,font=('arial',15),width = 650 ,height = 400)
output_summary.grid(row=7,column=10)

score_btn = customtkinter.CTkButton(master, text = '0%',font = ('arial',14,'bold'),width = 20 ,height = 45,corner_radius = 16,bg_color = "#F8F8F8",
                                    fg_color = "#bc5a45", hover_color = "#bc5a45")
score_btn.place(relx=0.55, rely=0.51, anchor=tkinter.CENTER)


def download_summary():
    res_summary = output_summary.get("0.0", "end")
    filename_path = "E:\\Research Papers\\TextSummary\\BBC News Summary\\GeneratedSummaries" + "\\" + category + "\\" + article_filename 
    #print("article_filename {} \nres_summary {} \filename_path {}".format(article_filename,res_summary,filename_path))

    file = open(filename_path,"w+")
    file.write(res_summary)
    file.close()
        
        

img = tkinter.PhotoImage(file="download.png")
download_btn = customtkinter.CTkButton(master, text = "", image = img,width = 4,height = 4,corner_radius = 5,bg_color = "#F8F8F8",fg_color = "#F8F8F8",
                                       hover_color = "#F8F8F8",command = download_summary)
download_btn.place(relx=0.94, rely=0.51, anchor=tkinter.CENTER)

    

# Function to open a file and populate text_box_A
def open_file():
    global category
    global article_filename
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    summary_path = file_path.split('/')
    category = summary_path[-2].strip()
    article_filename = summary_path[-1].strip()
    summary_path[4] = 'Summaries'
    summary_path = "/".join(summary_path)
    global title
    global article
    global summary
    if file_path:
        with open(file_path, "r") as file:
            title = next(file) # title of the doc
            article = file.read()
            input_text.delete("1.0", customtkinter.END)
            input_text.insert(customtkinter.END, article)
    if summary_path:
        with open(summary_path,"r") as file2:
            summary = file2.read()
   
    print("title {} \n article {} \n summary {}".format(title,article,summary))
 

# GET SUMMARIES OF HIGHEST F1 SCORE
def get_summaries():
    tf_idf_summary = tf_idf.get_input_text(article)
    print("\n\n TFIDF")
    tf_idf_acc = check_acc.calculate_f1_score(tf_idf_summary,summary)
    print("\n\n Glove")
    glove_summary = glove.get_input_text2(article)
    glove_acc = check_acc.calculate_f1_score(glove_summary,summary)
    print("\n\n TextRank")
    textrank_summary = textrank.get_input_text3(article)
    textrank_acc = check_acc.calculate_f1_score(textrank_summary,summary)
    
    if((tf_idf_acc > glove_acc ) and (tf_idf_acc > textrank_acc)):
        
        output_summary.delete("1.0", customtkinter.END)
        output_summary.insert(customtkinter.END,tf_idf_summary)
        score_btn.configure(text = str(tf_idf_acc)+"%")
        
    elif((glove_acc > tf_idf_acc) and (glove_acc > textrank_acc)):

        output_summary.delete("1.0", customtkinter.END)
        output_summary.insert(customtkinter.END,glove_summary)
        score_btn.configure(text = str(glove_acc)+"%")
        
    elif((textrank_acc > tf_idf_acc ) and (textrank_acc > glove_acc )):
        
        output_summary.delete("1.0", customtkinter.END)
        output_summary.insert(customtkinter.END,textrank_summary)
        score_btn.configure(text = str(textrank_acc)+"%")

    else:
        
        output_summary.delete("1.0", customtkinter.END)
        output_summary.insert(customtkinter.END,tf_idf_summary) # default summary method to show
        score_btn.configure(text = str(tf_idf_acc)+"%")
    


radio_var = tkinter.IntVar(value=0)

# GET SUMMARIES BASED ON THE METHOD SELETECD
def Get_Method():
    if (radio_var.get() == 1):
        
        tf_idf_summary = tf_idf.get_input_text(article)
        tf_idf_acc = check_acc.calculate_f1_score(tf_idf_summary,summary)
        output_summary.delete("1.0", customtkinter.END)
        output_summary.insert(customtkinter.END,tf_idf_summary)
        score_btn.configure(text = str(tf_idf_acc)+"%")
        
    elif(radio_var.get() == 2):
        glove_summary = glove.get_input_text2(article)
        glove_acc = check_acc.calculate_f1_score(glove_summary,summary)
        output_summary.delete("1.0", customtkinter.END)
        output_summary.insert(customtkinter.END,glove_summary)
        score_btn.configure(text = str(glove_acc)+"%")
        
    elif(radio_var.get() == 3):
        textrank_summary = textrank.get_input_text3(article)
        textrank_acc = check_acc.calculate_f1_score(textrank_summary,summary)
        output_summary.delete("1.0", customtkinter.END)
        output_summary.insert(customtkinter.END,textrank_summary)
        score_btn.configure(text = str(textrank_acc)+"%")

        
    else:
        output_summary.delete("1.0", customtkinter.END)
        output_summary.insert(customtkinter.END,"No summaries")
        score = '0'
       
   

# Use CTkButton instead of tkinter Button
upload_btn = customtkinter.CTkButton(master, text="Upload File",font = ('arial',15,),width = 40,height = 43,
                                     fg_color = "#36486b",hover_color = "#36486b",command = open_file)
upload_btn.grid(row = 10, column = 8,pady = 10)

summary_btn = customtkinter.CTkButton(master, text="Get Summary",font = ('arial',15),width = 40,height = 43,
                                      fg_color = "#587e76", hover_color = "#587e76",command = get_summaries)
summary_btn.grid(row = 10, column = 10)


label_radio_group = customtkinter.CTkLabel(master, text="Select Summary Method",font = ('arial',15))
label_radio_group.place(relx=0.17, rely=0.735)
radio_button_1 = customtkinter.CTkRadioButton(master, variable=radio_var, value = 1,text = "TF-IDF",fg_color="#bd5734",hover_color = "#bd5734",command = Get_Method)
radio_button_1.place(relx=0.3, rely=0.74)
radio_button_2 = customtkinter.CTkRadioButton(master, variable=radio_var, value = 2,text = "Glove",fg_color="#bd5734",hover_color = "#bd5734",command = Get_Method)
radio_button_2.place(relx=0.37, rely=0.74)
radio_button_3 = customtkinter.CTkRadioButton(master, variable=radio_var, value = 3,text = "TextRank",fg_color="#bd5734",hover_color = "#bd5734",command = Get_Method)
radio_button_3.place(relx=0.44, rely=0.74)

master.mainloop()

