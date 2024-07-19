"Name       : Roi Halali & Dor Kershberg  "
from tkinter import messagebox

from PIL import ImageTk

"Titel      : TeraSpeech aplication      "
"Sub Titel  : tkinter aplication          "

from face_landmark import *
from Data_Base import *
from Net import *
import glob
from scipy.io.wavfile import write
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import pandas as pd
import winsound
from googli import *
import os
from tkvideo import tkvideo
from calibrate import *
from PIL import ImageTk, Image


def labli(window,canvas,text ='כל הכבוד'):

    labeli=canvas.create_text(
        120, 70,
        text=text,
        fill="#00b2ff",
        font=("Inter-SemiBold", int(15.0)))
    # main_frame.winfo_children()[0].text.lift()
    window.after(2000, lambda :canvas.itemconfig(labeli, text=""))

def on_closing(user_id, recipt):
    if messagebox.askokcancel('האם אתה רוצה לצאת?','יציאה'):
        if (user_id != None):
            obj = DriveAPI()
            obj.FileUpdate(user_id, str(recipt)+'.csv', str(recipt)+'.csv')  #update of user progress
            sd.stop()
            os.remove("thera_speech_database.csv")
            os.remove(str(recipt)+'.csv')
        window.destroy()

def temp_entry(text, entry):
    def temp_text(e):
        current = entry.get()
        if current == text:
            entry.delete(0,"end")
        elif current == "":
            entry.insert(0, text)

    
    entry.bind("<FocusIn>", temp_text)
    entry.bind("<FocusOut>", temp_text)

def get_var_image():
    data_dir = 'tounge/'
    means = np.zeros(3)
    stds = np.zeros(3)

    i=0
    for phoneme_file in glob.iglob(data_dir + '**', recursive=True):
        if phoneme_file[-3:] == 'jpg':  # check npy file's form lest three letters
            image = Image.open(phoneme_file)
            pixels = np.asarray(image)
            # convert from integers to floats
            pixels = pixels.astype('float32')
            # calculate per-channel means and standard deviations
            mean = pixels.mean(axis=(0,1), dtype='float64')
            std = pixels.std(axis=(0,1), dtype='float64')
            means += mean
            stds += std
            
            i+=1
            print (i)
    return means/i, stds/i

def btn_clicked():
    print("Button Clicked")

def get_audio(win, path):
    # Sampling frequency
    freq = 16000

    # Recording duration
    duration = 5

    winsound.Beep(1000, 500)

    # Start recorder with the given values
    # of duration and sample frequency
    recording = sd.rec(int(duration * freq),
                       samplerate=freq, channels=1)

    # Record audio for the given number of seconds
    sd.wait()

    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    i = 1
    write(os.path.join(path, 'recording-' + str(i) + '.wav'), freq, recording)

    # Convert the NumPy array to audio file
    # wv.write(os.path.join(path, 'recording-' + str(i) + '.wav'), recording, freq, sampwidth=2)

    return recording

def check_fields(main_frame,canvas):
    
    df = pd.read_csv('thera_speech_database.csv')
    df1=[]
    Flag = False
    for idx, widget in enumerate(main_frame.winfo_children()):
        if isinstance(widget,Entry):
            df1.append(widget.get())

    for name, recipt1, password  in df.values :   #check if user in data base
        if df1[0]== str(recipt1) and df1[1]==password :
            Flag = True
            username=name
            recipt = recipt1

    if Flag == True :
        # check_csv(username,x)
        obj = DriveAPI()          #connect to google drive 
        user_id = obj.findId(df1[0]+'.csv')
        obj.FileDownload(user_id, df1[0]+'.csv')   #import the user file
        menu(main_frame,username, recipt, user_id)


    else:
        # Error massage:
        canvas.create_text(
            740, 360,
            text="שם משתמש או סיסמא אינם קיימים",
            fill="#ff7171",
            font=("Inter-SemiBold", int(8.0)))

def home_page(main_frame):

    # clear frame:
    main_frame.destroy()
    main_frame = Frame(window)
    main_frame.pack(fill=BOTH, expand=1)


    # Adding canvas:
    # The Canvas is a rectangular area intended for drawing pictures or other complex layouts.
    # You can place graphics, text, widgets or frames on a Canvas.
    canvas = Canvas(
        main_frame,
        bg = "#ffffff",
        height = 600,
        width = 1000,
        bd = 0,
        highlightthickness = 0,
        relief = "ridge")
    canvas.place(x = 0, y = 0)          # the canvas has to set in (0,0)

    # Background photo:
    # for using images -> PhotoImage object
    background_img = PhotoImage(file=f"background_home_page.png")
    background = canvas.create_image(
        502.5, 282.0,
        image=background_img)

    # page Title
    canvas.create_text(
        748.0, 70.5,
        text="Tera Speech",
        fill="#00b2ff",
        font=("Inter-SemiBold", int(45.0)))


    # TextBox- user name:
    canvas.create_text(
        806.0, 166.0,
        text="מספר קבלה",
        fill="#000000",
        font=("Inter-SemiBold", int(12.0)))

    entry0_img = PhotoImage(file=f"img_textBox0_home_page.png")
    entry0_bg = canvas.create_image(
        745.0, 200.0,
        image=entry0_img)

    entry0 = Entry(
        main_frame,
        bd=0,
        bg="#e4e3e3",
        justify=RIGHT,
        highlightthickness=0)

    entry0.place(
        x=655.0, y=183,
        width=180.0,
        height=32)


    # TextBox - password:
    canvas.create_text(
        806.0, 271.0,
        text="סיסמא",
        fill="#000000",
        font=("Inter-SemiBold", int(12.0)))

    entry1_img = PhotoImage(file=f"img_textBox1_home_page.png")
    entry1_bg = canvas.create_image(
        745.0, 305.0,
        image=entry1_img)

    entry1 = Entry(
        main_frame,
        bd=0,
        bg="#e4e3e3",
        justify=RIGHT,
        highlightthickness=0)

    entry1.place(
        x=655.0, y=288,
        width=180.0,
        height=32)

    # Button- start
    img0 = PhotoImage(file=f"img0_home_page.png")
    b0 = Button(
        main_frame,
        image=img0,
        borderwidth=0,
        highlightthickness=0,
        command=lambda:check_fields(main_frame,canvas),
        relief="flat")

    b0.place(
        x=707, y=393,
        width=75,
        height=27)


    # Button- sign up
    img1 = PhotoImage(file = f"img1_home_page.png")
    b1 = Button(
        main_frame,
        image = img1,
        borderwidth = 0,
        highlightthickness = 0,
        command=lambda: SignUp(main_frame),
        relief = "flat")

    b1.place(
        x = 710, y = 523,
        width = 75,
        height = 27)
    
    window.protocol("WM_DELETE_WINDOW", lambda:on_closing(None, None))
    window.mainloop()

def save_fields(main_frame,canvas):
    df = pd.read_csv('thera_speech_database.csv')
    df2 = []

    # take regist information from every entery
    for idx, widget in enumerate(main_frame.winfo_children()):
        if isinstance(widget, Entry):
            df2.append(widget.get())

    Flag = False
    if df2[1] not in df.values:
        Flag = True
    
    
    #check password & exist user
    if df2[2]==df2[3] and Flag==True and (len(df2[0].split()) ==2) :
        df2.pop()
        df2 = pd.Series(df2, index = df.columns)
        df= df.append(df2, ignore_index=True)
        user_prog = pd.DataFrame(columns=['date','move','range','time'])
        user_prog.to_csv(df2[1]+'.csv', index=False)
        df.to_csv("thera_speech_database.csv",index=False)
        obj = DriveAPI()
        obj.FileUpdate(thera_id, 'thera_speech_database', "thera_speech_database.csv" )
        obj.FileUpload(df2[1]+'.csv', file_id_users_dir)

        home_page(main_frame)

    else:
        # Error massage:
        canvas.create_text(
            494.0, 447.5,
            text="הכנס שוב סיסמא או מספר קבלה קיים במערכת",
            fill="#ff7171",
            font=("Inter-SemiBold", int(8.0)))

def SignUp(main_frame):

    # clear frame:
    main_frame.destroy()
    main_frame=Frame(window)
    main_frame.pack(fill=BOTH, expand=1)

    # Background:
    canvas = Canvas(
        main_frame,
        bg="#FFFFFF",
        height=600,
        width=1000,
        bd=0,
        highlightthickness=0,
        relief="ridge")
    canvas.place(x=0, y=0)

    background_img = PhotoImage(file=f"background_signup.png")
    background = canvas.create_image(
        502, 300,
        image=background_img)


    # Title
    canvas.create_text(
        500.0, 78.0,
        text="הרשמה",
        fill="#00b2ff",
        font=("Inter-SemiBold", int(45.0)))


    # TextBox- Full name
    entry0_img = PhotoImage(file=f"img_textBox0_signup.png")
    entry0_bg = canvas.create_image(
        499.5, 161.0,
        image=entry0_img)

    entry0 = Entry(
        main_frame,
        bd=0,
        justify=RIGHT,
        bg="#e4e3e3",
        highlightthickness=0)

    entry0.place(
        x=397.0, y=141,
        width=205.0,
        height=38)
    
    entry0.insert(0, 'שם מלא')
    temp_entry('שם מלא', entry0)
    

    # TextBox- Receipt number
    entry1_img = PhotoImage(file=f"img_textBox1_signup.png")
    entry1_bg = canvas.create_image(
        499.5, 241.0,
        image=entry1_img)

    entry1 = Entry(
        main_frame,
        bd=0,
        justify=RIGHT,
        bg="#e4e3e3",
        highlightthickness=0)

    entry1.place(
        x=397.0, y=221,
        width=205.0,
        height=38)
    entry1.insert(0, 'מספר קבלה' )
    
    temp_entry('מספר קבלה' , entry1)


    # TextBox- password
    entry2_img = PhotoImage(file=f"img_textBox2_signup.png")
    entry2_bg = canvas.create_image(
        499.5, 320.0,
        image=entry2_img)

    entry2 = Entry(
        main_frame,
        bd=0,
        justify=RIGHT,
        bg="#e4e3e3",
        highlightthickness=0)

    entry2.place(
        x=397.0, y=300,
        width=205.0,
        height=38)
    entry2.insert(0, 'סיסמא')

    temp_entry('סיסמא' , entry2)


    # TextBox- confirm password
    entry3_img = PhotoImage(file=f"img_textBox3_signup.png")
    entry3_bg = canvas.create_image(
        499.5, 394.0,
        image=entry3_img)

    entry3 = Entry(
        main_frame,
        bd=0,
        justify=RIGHT,
        bg="#e4e3e3",
        highlightthickness=0)

    entry3.place(
        x=397.0, y=374,
        width=205.0,
        height=38)
    entry3.insert(0, 'אימות סיסמא')
    
    temp_entry('אימות סיסמא' , entry3)


    # button - submit
    img0 = PhotoImage(file=f"img0_signup.png")
    b0 = Button(
        main_frame,
        image=img0,
        borderwidth=0,
        highlightthickness=0,
        command=lambda:save_fields(main_frame,canvas),
        relief="flat")

    b0.place(
        x=459, y=481,
        width=75,
        height=27)


    # Button - back
    img1 = PhotoImage(file=f"back.png")
    b1 = Button(
        main_frame,
        image=img1,
        borderwidth=0,
        highlightthickness=0,
        command= lambda: home_page(main_frame),
        relief="flat")

    b1.place(
        x=735, y=530,
        width=33,
        height=30)
    
    window.protocol("WM_DELETE_WINDOW", lambda:on_closing(None, None))
    window.resizable(False, False)
    window.mainloop()

def nothing(x):
    pass

def find_best_mask():
    # Capture web camera video
    cap = cv2.VideoCapture(1)
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

    # Creating a window for later use
    cv2.namedWindow('best mask', cv2.WINDOW_AUTOSIZE)

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv2.createTrackbar('H_Low', 'best mask', 0, 20, nothing)
    cv2.createTrackbar('H_High', 'best mask', 0, 20, nothing)
    cv2.createTrackbar('S', 'best mask', 0, 255, nothing)
    cv2.createTrackbar('V', 'best mask', 0, 255, nothing)
    cv2.createTrackbar('blur', 'best mask', 1, 25, nothing)

    # Set default value for Max HSV trackbars
    cv2.setTrackbarPos('H_Low', 'best mask', 20)
    cv2.setTrackbarPos('H_High', 'best mask', 20)
    cv2.setTrackbarPos('S', 'best mask', 255)
    cv2.setTrackbarPos('V', 'best mask', 255)
    cv2.setTrackbarPos('blur', 'best mask', 25)

    while (1):

        _, frame = cap.read()
        frame = cv2.resize(frame, (428,354))

        blurV = int(cv2.getTrackbarPos('blur', 'best mask'))
        blur = cv2.blur(frame, (blurV, blurV))  # blur for decrease noise
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Get current positions of all trackbars
        h_lower = cv2.getTrackbarPos('H_Low', 'best mask')
        h_upper = cv2.getTrackbarPos('H_High', 'best mask')

        s = cv2.getTrackbarPos('S', 'best mask')
        v = cv2.getTrackbarPos('V', 'best mask')

        # Set minimum and maximum HSV values to display
        lower_lr = np.array([0, 0, 0])
        upper_lr = np.array([h_lower, s, v])
        lr_mask = cv2.inRange(hsv, lower_lr, upper_lr)  # Low red threshold image.

        lower_ur = np.array([179 - 20, 0, 0])
        upper_ur = np.array([179 - h_upper, s, v])
        ur_mask = cv2.inRange(hsv, lower_ur, upper_ur)  # High red threshold image.

        mask_tune = lr_mask + ur_mask
        mask_type=True

        # Convert to HSV format and color threshold
        result = cv2.bitwise_and(frame, frame, mask=mask_tune)

        # Display result image
        cv2.imshow('best mask', result)

        # display morphologic results.
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)  # from BGR to gray

        # elipses detection
        x = int(428 * 0.07)
        y = int(354 * 0.1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (x, y)) * 255
        elipse = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
        edges = cv2.Canny(elipse, 75, 170)  # Canny Edge Detection
        cv2.imshow('2', edges)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            # save parameters
            np.save("mask/mask_type.npy", mask_type, allow_pickle=True)
            np.save("mask/mask_tune.npy", mask_tune, allow_pickle=True)
            cv2.destroyAllWindows()
            cap.release()
            break

def menu(main_frame,username, recipt, user_id):
    
    sd.stop()
    # clear frame:
    main_frame.destroy()
    main_frame = Frame(window)
    main_frame.pack(fill=BOTH, expand=1)

    # Background:
    canvas = Canvas(
        main_frame,
        bg="#ffffff",
        height=600,
        width=1000,
        bd=0,
        highlightthickness=0,
        relief="ridge")
    canvas.place(x=0, y=0)

    background_img = PhotoImage(file=f"background_menu.png")
    background = canvas.create_image(
        500.0, 300.0,
        image=background_img)

    # Page Title
    canvas.create_text(
        499.5, 66.0,
        text="שלום "+username.split()[0],
        fill="#509cf6",
        font=("IstokWeb-Regular", int(30.0)))

    # Page front text
    canvas.create_text(
        499.5, 150, 
        text="ברוך הבא למתחם האימונים",
        fill="#000000",
        font=("IstokWeb-Regular", int(25.0)))


    #face motion Button
    img0 = PhotoImage(file=f"img0_menu.png")
    b0 = Button(
        main_frame,
        image=img0,
        borderwidth=0,
        highlightthickness=0,
        command=lambda:face_motion_page(main_frame,username, recipt, user_id),
        relief="flat")

    b0.place(
        x=169, y=211,
        width=187,
        height=177)

    # speech Button
    img1 = PhotoImage(file=f"img1_menu.png")
    b1 = Button(
        main_frame,
        image=img1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: speech_page(main_frame, username, recipt, user_id),
        relief="flat")

    b1.place(
        x=402, y=211,
        width=195,
        height=177)

    # Button-progress page
    img2 = PhotoImage(file=f"img2_menu.png")
    b2 = Button(
        main_frame,
        image=img2,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: results_page(main_frame, username, recipt, user_id),
        relief="flat")

    b2.place(
        x=643, y=211,
        width=195,
        height=177)

    # Button back
    img3 = PhotoImage(file=f"back.png")
    b3 = Button(
        main_frame,
        image=img3,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: home_page(main_frame),
        relief="flat")

    b3.place(
        x=805, y=531,
        width=33,
        height=30)

    img4 = PhotoImage(file=f"camera_tune.png")
    b4 = Button(
        image=img4,
        borderwidth=0,
        highlightthickness=0,
        command=lambda :find_best_mask(),
        relief="flat")

    b4.place(
        x=169, y=527,
        width=111,
        height=37)
    
    window.protocol("WM_DELETE_WINDOW", lambda:on_closing(user_id, recipt))
    window.resizable(False, False)
    window.mainloop()

def face_motion_page(main_frame, username, recipt, user_id):

    def show_motion(main_frame,motion,new_button,title,text,video_path):

        # Pages setings:
        #open_mouth->smile
        if motion == "open_mouth":
            # delete the start button
            for idx, widget in enumerate(main_frame.winfo_children()):
                if isinstance(widget, Button) and idx == 3:
                    widget.destroy()

            # Next page Definitions
            next_motion = "smile"
            next_text = "תרגול תנועת חיוך"
            next_video_path = "videos/smile.mp4"    ## temporary assign with tounge_down
            new_button.configure(command=lambda: show_motion(main_frame,next_motion,new_button,title,next_text,next_video_path))

        # smile->kiss
        elif motion == "smile":
            # Next page Definitions
            next_motion = "kiss"
            next_text = "תרגול תנועת נשיקה"
            next_video_path = "videos\kiss.mp4"
            new_button.configure(
            command=lambda: show_motion(main_frame, next_motion, new_button, title, next_text, next_video_path))

        # kiss->tounge_down
        elif motion == "kiss":
            # Next page Definitions
            next_motion = "down"
            next_text = "תרגול לשון למטה"
            next_video_path = "videos/tounge_down.mp4"
            new_button.configure(
            command=lambda: show_motion(main_frame, next_motion, new_button, title, next_text, next_video_path))

        # tounge_down->tounge_up
        elif motion == "down":
            # Next page Definitions
            next_motion = "up"
            next_text = "תרגול לשון למעלה"
            next_video_path = "videos/tounge_up.mp4"
            new_button.configure(
            command=lambda: show_motion(main_frame, next_motion, new_button, title, next_text, next_video_path))

        # tounge_up->tounge_left
        elif motion == "up":
            # Next page Definitions
            next_motion = "left"
            next_text = "תרגול לשון שמאלה"
            next_video_path = "videos/tounge_left.mp4"
            new_button.configure(
                command=lambda: show_motion(main_frame, next_motion, new_button, title, next_text, next_video_path))

        # tounge_left->tounge_right
        elif motion == "left":
            # Next page Definitions
            next_motion = "right"
            next_text = "תרגול לשון ימינה"
            next_video_path = "videos/tounge_right.mp4"
            new_button.configure(
                command=lambda: show_motion(main_frame, next_motion, new_button, title, next_text, next_video_path))

        # tounge_right->finel
        elif motion == "right":
            # Next page Definitions
            next_motion = "finel"
            next_text = "כל הכבוד! סיימת את התרגול בהצלחה"
            next_video_path = "videos\smile.mp4"
            new_button.configure(
                command=lambda: show_motion(main_frame, next_motion, new_button, title, next_text, next_video_path))

        # finel
        elif motion == "finel":
            # Next page Definitions
            new_button.configure(command=lambda: (main_frame,username, recipt, user_id))

        sd.stop()  # stop the sound

        # Title
        canvas.itemconfig(title, text=text)

        # Explanation Video:
        # video lable
        video_Label = Label(main_frame)
        video_Label.place(
            x=524, y=87,
            width=428,
            height=354)

        # Video
        # cap_vid = cv2.VideoCapture(video_path)
        # fps = cap_vid.get(cv2.CAP_PROP_FPS)

        player = tkvideo(video_path, video_Label, hz = 31.5)
        player.play()

        # Create a Label for camera video
        label = Label(main_frame)
        label.place(
            x=72, y=87,
            width=428,
            height=354)

        # Define function to show frame
        overall_time = 0

        reps = 0
        start = time.time()
        old_state = 'close'
        new_state = 'close'

        # Capture web camera video
        cap = cv2.VideoCapture(1)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
        prog = pd.read_csv(str(recipt) + '.csv')
        def show_frames():
            nonlocal start
            nonlocal old_state
            nonlocal new_state
            nonlocal prog
            nonlocal reps
            
            date_time = datetime.now()
            df = []

            img = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
            face_width = face_data(img)
            
            if check_if_good(img, face_width) == 'good' and face_width != 0:
                vert, hor = vertical_horizontal(img)
                # Get the latest frame and convert into Image
                if (vert and hor != 0):
                        new_state, distance, reps = face(img, vert, hor, motion, reps)
                        # print (reps)

                if new_state == "open" and old_state == "close" :
                    
                    if reps == 2 or motion == 'smile' or motion =='open_mouth' or motion == 'kiss':
                        end = time.time()
                        overall_time = end - start
                        df.extend([date_time, motion , round(distance, 2), round(overall_time, 2)])
                        df = pd.Series(df, index=prog.columns)
                        prog = prog.append(df, ignore_index=True)
                        prog.to_csv(str(recipt) + '.csv', index=False)
        
                        labli(main_frame,canvas)
                        print(overall_time)
                        old_state = "open"
                        reps = 0

    
                elif new_state == "close" and old_state == "open":
                    overall_time = 0
                    start = time.time()
                    old_state = "close"
            elif check_if_good(img, face_width) == 'too_close' and face_width != 0:
                labli(main_frame,canvas, 'התרחק')
            elif check_if_good(img, face_width) == 'too_far' and face_width != 0:
                labli(main_frame,canvas, 'התקרב')
            
            # Get the latest frame and convert into Image
            image = Image.fromarray(img)
            # Convert image to PhotoImage
            imgtk = ImageTk.PhotoImage(image=image)
            label.imgtk = imgtk
            label.configure(image=imgtk)

            # Repeat after an interval to capture continiously
            label.after(20, show_frames)

        show_frames()
        window.mainloop()
        cap.release()
        cv2.destroyAllWindows()

    # clear frame:
    main_frame.destroy()
    main_frame = Frame(window)
    main_frame.pack(fill=BOTH, expand=1)

    # Background:
    canvas = Canvas(
        main_frame,
        bg="#ffffff",
        height=600,
        width=1000,
        bd=0,
        highlightthickness=0,
        relief="ridge")
    canvas.place(x=0, y=0)

    background_img = PhotoImage(file=f"background_face.png")
    background = canvas.create_image(
        491.0, 306.0,
        image=background_img)

    #Title
    title=canvas.create_text(
        490.5, 50.5,
        text="אימון תנועות פנים",
        fill="#00b2ff",
        font=("IstokWeb-Regular", int(30.0)))


    # Button- back to menu
    img0 = PhotoImage(file=f"back.png")
    b0 = Button(
        main_frame,
        image=img0,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: menu(main_frame, username, recipt, user_id),
        relief="flat")

    b0.place(
        x=910, y=539,
        width=33,
        height=30)


    # Button- next
    img2 = PhotoImage(file=f"img2_face.png")
    b2 = Button(
        main_frame,
        image=img2,
        borderwidth=0,
        highlightthickness=0,
        command=btn_clicked,
        relief="flat")

    b2.place(
        x = 346, y = 454,
        width=111,
        height=37)


    # Button- start
    img1 = PhotoImage(file=f"img1_face.png")
    b1 = Button(
        main_frame,
        image=img1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: show_motion(main_frame,"open_mouth",b2,title,"אימון פתיחת פה","videos/open_mouth.mp4"))

    b1.place(
        x=200, y=454,
        width=111,
        height=37)


    # right rectangle
    canvas.create_rectangle(
        524, 87, 524 + 428, 87 + 354,
        fill="#e8e8ef",
        outline="")

    # left rectangle
    canvas.create_rectangle(
        72, 87, 72 + 428, 87 + 354,
        fill="#e8e8ef",
        outline="")
    
    # Intro Text
    im=cv2.imread("face_motion_introtext.png")
    im=cv2.resize(im,(428,354))

    img3 = PhotoImage(file=f"face_motion_introtext.png")

    b3 = Label(
            main_frame,
            image=img3,
            borderwidth=0,
            highlightthickness=0,
            relief="flat")
    
    b3.place(
            x=72, y=87,
            width=428,
            height=354)

    # video lable
    video_Label = Label(main_frame)
    video_Label.place(
        x=524, y=87,
        width=428,
        height=354)

    # cap_vid = cv2.VideoCapture("videos/introface.mp4")
    vid_bg = "videos/music_sub/introface.wav" 
    var = "אימון הבעות פנים"
    # show_motion(main_frame,None,b2,title,var,"videos/introface.mp4","videos/music_sub/introface.wav")
    player = tkvideo("videos/introface.mp4", video_Label, hz = 31.5)
    player.play()
    data, fs = sf.read(vid_bg, dtype='float32')
    sd.play(data, fs)

    window.protocol("WM_DELETE_WINDOW", lambda:on_closing(user_id, recipt))
    window.resizable(False, False)
    window.mainloop()

def speech_page(main_frame, username, recipt, user_id):
    #global virebels:
    start=0
    end=0
    succses=False
    flag = True

    thread_array=[]

    def record_acurecy(phoneme):
        
        nonlocal flag
        nonlocal start
        nonlocal end
        nonlocal succses


        if flag == True:
            start = time.time()
            flag = False
        
        # Files dirs:
        records = 'audio/records/'
        rec_proc = 'audio/records processed/'

        # clear Files:
        shutil.rmtree(records)
        shutil.rmtree(rec_proc)
        os.makedirs(records)
        os.makedirs(rec_proc)

        # Recording from computer:
        # Sampling frequency
        freq = 16000

        # Recording duration
        duration = 3

        winsound.Beep(500, 500)
        
        # Start recorder with the given values
        # of duration and sample frequency
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
        sd.wait()
        # This will convert the NumPy array to an audio
        # file with the given sampling frequency
        write(os.path.join(records, 'recording-'  + '.wav'), freq, recording)
        
        # Pre processing the new record:
        prepere_new_input(records, rec_proc)
        record_acurecy = new_rec_pred(rec_proc, phoneme)
        
        if record_acurecy > 0.6:
            succses=True
            end=time.time()
            print("succses")
            
            
        else:
            sucsses=False
            print("Fail")

    def phoneme_train(main_frame, next_button, record_button,title ,phoneme):
                
        nonlocal flag
        nonlocal vid
        img4 = PhotoImage(file=f"img3_face.png")

        for idx, widget in enumerate(main_frame.winfo_children()):
            if isinstance(widget, Button) and idx == 3:
                widget.configure(image = img4)

        
        # Set titles and phonemes:
        sd.stop()

        # aa -> ae :
        if (phoneme == aa).all():
            canvas.itemconfig(title, text="תרגול תנועת")
            bg = PhotoImage(file=f"background_a.png")
            canvas.itemconfig(background,image=bg)
            video_name = "videos/aa.mp4"  # This is your video file path
            audio_name = "videos/music_sub/aa.wav"
            next_button.configure(command=lambda:phoneme_train(main_frame, next_button, record_button, title, eh))
            record_button.configure(command=lambda: record_acurecy(phoneme))

        # eh -> iy :
        elif (phoneme == eh).all():
            bg = PhotoImage(file=f"background_e.png")
            canvas.itemconfig(background, image=bg)
            video_name = "videos/eh.mp4"  # This is your video file path
            audio_name = "videos/music_sub/eh.wav"
            next_button.configure(command=lambda:[phoneme_train(main_frame, next_button, record_button, title, iy)])
            record_button.configure(command=lambda: record_acurecy(phoneme))

        # iy -> ow :
        elif (phoneme == iy).all():
            bg = PhotoImage(file=f"background_i.png")
            canvas.itemconfig(background, image=bg)
            video_name = "videos/iy.mp4"  # This is your video file path
            audio_name = "videos/music_sub/iy.wav"
            next_button.configure(command=lambda:[phoneme_train(main_frame, next_button, record_button, title, ow)])
            record_button.configure(command=lambda: record_acurecy(phoneme))

        # ow -> uw :
        elif (phoneme == ow).all():
            bg = PhotoImage(file=f"background_o.png")
            canvas.itemconfig(background, image=bg)
            video_name = "videos/ow.mp4"  # This is your video file path
            audio_name = "videos/music_sub/ow.wav"
            next_button.configure(command=lambda:[phoneme_train(main_frame, next_button, record_button, title, uw)])
            record_button.configure(command=lambda: record_acurecy(phoneme))

        # uw -> finish :
        elif (phoneme == uw).all():
            bg = PhotoImage(file=f"background_u.png")
            canvas.itemconfig(background, image=bg)
            video_name = "videos/wuw.mp4"  # This is your video file path
            audio_name = "videos/music_sub/wuw.wav"
            next_button.configure(command=lambda: phoneme_train(main_frame, next_button, record_button, title, non))
            record_button.configure(command=lambda: record_acurecy(phoneme))

        # finish:
        elif (phoneme == non).all():
            canvas.itemconfig(title, text="כל הכבוד! סיימת את האימון")
            bg = PhotoImage(file=f"background_face.png")
            canvas.itemconfig(background, image=bg)
            bg = PhotoImage(file=f"background_o.png")
            canvas.itemconfig(background, image=bg)
            video_name = "mouth_open.mp4"  # This is your video file path
            next_button.configure(command=lambda: menu(main_frame,username))
            record_button.configure(command=lambda: btn_clicked())


        video_label = Label(main_frame)
        video_label.place(
            x=524, y=87,
            width=428,
            height=354)

        # Play video background sound
        data, fs = sf.read(audio_name, dtype='float32')
        sd.play(data, fs)


        player = tkvideo(video_name, video_label,loop = 0 ,size = (428, 354), hz = 31.5)
        vid = player.play()

        # Capture web camera video
        cap = cv2.VideoCapture(1)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
        ret,_=cap.read()

        # Create a Label for camera video
        label = Label(main_frame)
        label.place(
            x=72, y=87,
            width=428,
            height=354)
        

        # Define function to show frame
        overall_time = 0
        start = time.time()

        prog = pd.read_csv(str(recipt)+'.csv')

        # Define function to show frame
        def show_frames():
            # nonlocal buttonClicked
            nonlocal succses
            nonlocal start
            nonlocal end
            nonlocal flag
            nonlocal prog
            nonlocal ret
            nonlocal phoneme
            
            # if buttonClicked:
            #     return
                   

            df=[]
            date_time=datetime.now()

            if succses==True:
                overall_time=end-start
                print(overall_time)
                flag = True
                start = time.time()

                if (phoneme == aa).any():
                    phoneme_name = 'aa'
                if (phoneme == eh).any():
                    phoneme_name = 'ae'
                if (phoneme == iy).any():
                    phoneme_name = 'iy'
                if (phoneme == ow).any():
                    phoneme_name = 'ow'
                if (phoneme == uw).any():
                    phoneme_name = 'uw'
                    
                df.extend([date_time, phoneme_name, 50, overall_time])
                df = pd.Series(df, index = prog.columns)
                prog = prog.append(df, ignore_index=True)
                prog.to_csv(str(recipt)+'.csv', index=False)
                succses = False
                labli(main_frame,canvas)
                
            
            elif ret == True:
                img = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)   
                # Get the latest frame and convert into Image
                image = Image.fromarray(img)
                # Convert image to PhotoImage
                imgtk = ImageTk.PhotoImage(image=image)
                label.imgtk = imgtk
                label.configure(image=imgtk)
                # Repeat after an interval to capture continiously
                
                label.after(20, show_frames)

        show_frames()
        window.mainloop()
        cap.release()
        cv2.destroyAllWindows()
        

    # clear frame:
    main_frame.destroy()
    main_frame = Frame(window)
    main_frame.pack(fill=BOTH, expand=1)

    # Background:
    canvas = Canvas(
        main_frame,
        bg="#ffffff",
        height=600,
        width=1000,
        bd=0,
        highlightthickness=0,
        relief="ridge")
    canvas.place(x=0, y=0)

    background_img = PhotoImage(file=f"background_face.png")
    background = canvas.create_image(
        491.0, 306.0,
        image=background_img)

    #Title
    title=canvas.create_text(
        505, 49,
        text="אימון דיבור",
        fill="#2A8CFF",
        font=("IstokWeb-Regular", int(30.0)))


    # Button- back to menu
    img0 = PhotoImage(file=f"back.png")
    b0 = Button(
        main_frame,
        image=img0,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: menu(main_frame,username, recipt, user_id),
        relief="flat")

    b0.place(
        x=910, y=539,
        width=33,
        height=30)

    # Button- next
    img2 = PhotoImage(file=f"img2_face.png")
    b2 = Button(
        main_frame,
        image=img2,
        borderwidth=0,
        highlightthickness=0,
        command=btn_clicked,
        relief="flat")

    b2.place(
        x=298, y=464,
        width=111,
        height=37)
    
    
    # Button- start
    img1 = PhotoImage(file=f"img1_face.png")
    b1 = Button(
        main_frame,
        image=img1,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: phoneme_train(main_frame, b2,b1,title ,phoneme=aa),
        relief="flat")

    b1.place(
        x=164, y=464,
        width=111,
        height=37)


    canvas.create_rectangle(
        72, 87, 72+428, 87+354,
        fill = "#e8e8ef",
        outline = "")

    canvas.create_rectangle(
        524, 87, 524 + 428, 87 + 354,
        fill="#e8e8ef",
        outline="")
    
    img3 = PhotoImage(file=f"speech_page_introtext.png")

    b3 = Label(
            main_frame,
            image=img3,
            borderwidth=0,
            highlightthickness=0,
            relief="flat")
    
    b3.place(
            x=72, y=87,
            width=428,
            height=354)
    
    video_label = Label(main_frame)
    video_label.place(
        x=524, y=87,
        width=428,
        height=354)
    
    # read video to display on label
    # cap = cv2.VideoCapture("videos\introsound.mp4")
    # fps = cap.get(cv2.CAP_PROP_FPS)
    player = tkvideo("videos\introsound.mp4", video_label, size = (428, 354), hz = 40.0)
    vid = player.play()

    # Play video background sound
    data, fs = sf.read('videos\music_sub\introsound.wav', dtype='float32')
    sd.play(data, fs)

    window.protocol("WM_DELETE_WINDOW", lambda:on_closing(user_id, recipt))
    window.resizable(False, False)
    window.mainloop()

def results_page(main_frame, username, recipt, user_id):
    # clear frame:
    main_frame.destroy()
    main_frame = Frame(window)
    main_frame.pack(fill=BOTH, expand=1)

    # Create A Canvas
    canvas = Canvas(
        main_frame,
        bg="#ffffff",
        height=600,
        width=1000,
        bd=0,
        highlightthickness=0,
        relief="ridge")
    canvas.place(x=0, y=0)

    background_img = PhotoImage(file=f"background_result.png")
    background = canvas.create_image(
        500.0, 300.0,
        image=background_img)

    # Title
    canvas.create_text(
        499.5, 64.0,
        text="תוצאות אימונים",
        fill="#00b2ff",
        font=("IstokWeb-Regular", int(40.0)))

    # Button- progress
    img1 = PhotoImage(file=f"img1_result.png")
    b1 = Button(
        main_frame,
        image=img1,
        borderwidth=0,
        highlightthickness=0,
        command= lambda:progress_page(main_frame,username, recipt, user_id),
        relief="flat")

    b1.place(
        x=348, y=485,
        width=304,
        height=52)


    # Button- back to menu
    img0 = PhotoImage(file=f"back.png")
    b0 = Button(
        main_frame,
        image=img0,
        borderwidth=0,
        highlightthickness=0,
        command=lambda: menu(main_frame, username, recipt, user_id),
        relief="flat")

    b0.place(
        x=805, y=531,
        width=33,
        height=30)


    # Create ANOTHER Frame INSIDE the Canvas
    second_frame = Frame(canvas)
    # Add that New frame To a Window In The Canvas
    canvas.create_window((283, 107), height=365, width=432, window=second_frame, anchor="nw")

    # Create A Canvas
    scroll_canvas = Canvas(
        second_frame,
        bg="#ffffff",
        height=365,
        width=432,
        bd=0,
        highlightthickness=0,
        relief="ridge")
    scroll_canvas.place(x=0, y=0)

    # Add A Scrollbar To The Canvas
    my_scrollbar = Scrollbar(second_frame, orient=VERTICAL, command=scroll_canvas.yview)
    my_scrollbar.pack(side=RIGHT, fill=Y)

    # Configure The Canvas
    scroll_canvas.configure(yscrollcommand=my_scrollbar.set)
    scroll_canvas.bind('<Configure>', lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))

    # Create ANOTHER Frame INSIDE the Canvas
    third_frame = Frame(scroll_canvas)

    # Add that New frame To a Window In The Canvas
    scroll_canvas.create_window((0, 0), window=third_frame, anchor="nw")

    df = pd.read_csv(str(recipt)+'.csv')
    df['Dates'] = pd.to_datetime(df['date']).dt.date
    y = df['Dates'].unique()
    df = df.set_index('Dates')
    df = df.drop(columns=['date'])
    button_dict = {}

    
    for i in (y):
        
        def action(x = i): 
            return daily_results_page(df.loc[x] ,main_frame, username, recipt, user_id)
        
        button_dict[i] =  Button(
            third_frame,
            width=32,
            height=3,
            text=i,
            bg="#D0D6F9",
            borderwidth=0,
            highlightthickness=0,
            command = action,
            relief="flat")

        button_dict[i].pack(side=BOTTOM, pady=6, padx=100)
        
    window.protocol("WM_DELETE_WINDOW", lambda:on_closing(user_id, recipt))
    window.mainloop()

def graph(data):
    
    folder = 'graphs'
    if len(os.listdir(folder)) != 0:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
                
    #count vs moves graphs
    y = data.groupby('move')['range'].count()    
    x = y.index
    
    plt.clf()
    plt.ion()
    plt.bar(x, y, color = ['cyan','skyblue','lightpink','brown','black','red', 'green', 'blue'])
    plt.xlabel("moves")
    plt.ylabel("count")
    plt.title("Number of returns per movement")
    plt.ioff()
    plt.savefig('graphs/img_a.png')

    #mean range vs moves graphs
    df = data.set_index('move')

    booli = False
    booli = 'aa' in data['move'].unique()
    if booli:
        y1 = df.drop("aa")
    booli = 'phoneme' in data['move'].unique()
    if booli:
        y1 = df.drop("phoneme")
    booli = 'iy' in data['move'].unique()
    if booli:
        y1 = df.drop("iy")
    booli = 'ow' in data['move'].unique()
    if booli:
        y1 = df.drop("ow")
    booli = 'uw' in data['move'].unique()
    if booli:
        y1 = df.drop("uw")

    y1 = df.groupby('move')['range'].mean()

    x = y1.index
    
    plt.clf()
    plt.ion()
    plt.bar(x, y1, color = ['cyan','skyblue','lightpink','brown','black','red', 'green', 'blue'])
    plt.xlabel("moves")
    plt.ylabel("average range")
    plt.title("Average range per movement")
    plt.ioff()
    plt.savefig('graphs/img_c.png')

    
    
    #total time vs moves graphs
    y2 = data.groupby('move')['time'].sum()
    x = y2.index
    plt.clf()
    plt.ion()
    plt.bar(x, y2, color = ['cyan','skyblue','lightpink','brown','black','red', 'green', 'blue'])
    plt.xlabel("moves")
    plt.ylabel("total time [sec]")
    plt.title("Total time per movement")
    plt.ioff()
    plt.savefig('graphs/img_b.png')
            
    
    #avg time vs moves graphs
    y3 = data.groupby('move')['time'].mean()
    x = y3.index

    plt.clf()
    plt.ion()
    plt.bar(x, y3, color = ['cyan','skyblue','lightpink','brown','black','red', 'green', 'blue'])
    plt.xlabel("moves")
    plt.ylabel("avg time [sec]")
    plt.title("Avg time per movement")
    plt.ioff()
    plt.savefig('graphs/img_d.png')
    
def graphs_in_time(data):
    folder = 'graphs'
    if len(os.listdir(folder)) != 0:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    # count vs moves graphs
    y = data.groupby('Dates')['range'].count()
    x = y.index

    plt.clf()
    plt.ion()
    plt.plot_date(x, y, linestyle='solid')
    plt.xlabel("time")
    plt.ylabel("count")
    plt.title("Number of returns per movement in time")
    plt.tick_params(axis='x', labelsize=10, rotation=90)
    plt.ioff()
    plt.savefig('graphs/img_a.png')

    # count vs moves graphs
    y1 = data.groupby('Dates')['range'].mean()
    x = y1.index

    plt.clf()
    plt.ion()
    plt.plot_date(x, y1, linestyle='solid')
    plt.xlabel("time")
    plt.ylabel("average range")
    plt.title("Average range per movement in time")
    plt.tick_params(axis='x', labelsize=10, rotation=90)
    plt.ioff()
    plt.savefig('graphs/img_c.png')

    # total time vs moves graphs
    y2 = data.groupby('Dates')['time'].sum()
    x = y2.index

    plt.clf()
    plt.ion()
    plt.plot_date(x, y2, linestyle='solid')
    plt.xlabel("time")
    plt.ylabel("total time [sec]")
    plt.title("Total time per movement in time")
    plt.tick_params(axis='x', labelsize=10, rotation=90)
    plt.ioff()
    plt.savefig('graphs/img_b.png')

    # avg time vs moves graphs
    y3 = data.groupby('Dates')['time'].mean()
    x = y3.index

    plt.clf()
    plt.ion()
    plt.plot_date(x, y3, linestyle='solid')
    plt.xlabel("time")
    plt.ylabel("avg time [sec]")
    plt.title("Avg time per movement in time")
    plt.tick_params(axis='x', labelsize=10, rotation=90)
    plt.ioff()
    plt.savefig('graphs/img_d.png')
        
def daily_results_page(data, main_frame, username, recipt, user_id):
    
    
    graph(data)
    img = cv2.imread('graphs/img_a.png')
    img_a = cv2.resize(img,(450,200))
    cv2.imwrite('graphs/img_a.png', img_a)

    
    img = cv2.imread('graphs/img_b.png')
    img_b = cv2.resize(img,(450,200))
    cv2.imwrite('graphs/img_b.png', img_b)

    
    img = cv2.imread('graphs/img_c.png')
    img_c = cv2.resize(img,(450,200))
    cv2.imwrite('graphs/img_c.png', img_c)

    
    img = cv2.imread('graphs/img_d.png')
    img_d = cv2.resize(img,(450,200))
    cv2.imwrite('graphs/img_d.png', img_d)

    # Clear frame:
    main_frame.destroy()
    main_frame = Frame(window)
    main_frame.pack(fill=BOTH, expand=1)

    canvas = Canvas(
        main_frame,
        bg="#ffffff",
        height=600,
        width=1000,
        bd=0,
        highlightthickness=0,
        relief="ridge")
    canvas.place(x=0, y=0)

    background_img = PhotoImage(file=f"background_dayli.png")
    background = canvas.create_image(
        500.0, 300.0,
        image=background_img)


    # Title
    canvas.create_text(
        500.0, 44.5,
        text = "תוצאות יומיות",
        fill = "#509cf6",
        font = ("IstokWeb-Regular", int(30.0)))


    # Charts:
    canvas.create_rectangle(
        515, 87, 515 + 450, 87 + 200,
        fill="#e8e8ef",
        outline="")
    
    img1 = PhotoImage(file=f'graphs\img_a.png')
    b1 = Button(
        main_frame,
        image=img1,
        borderwidth=0,
        highlightthickness=0,
        command=btn_clicked,
        relief="flat")

    b1.place(
        x=515, y=87,
        width=450,
        height=200)
    
    canvas.create_rectangle(
        36, 87, 36 + 450, 87 + 200,
        fill="#e8e8ef",
        outline="")
    
    img2 = PhotoImage(file=f'graphs\img_b.png')
    b2 = Button(
        main_frame,
        image=img2,
        borderwidth=0,
        highlightthickness=0,
        command=btn_clicked,
        relief="flat")
    
    b2.place(
        x=36, y=87,
        width=450,
        height=200)

    canvas.create_rectangle(
        36, 321, 36 + 450, 321 + 200,
        fill="#e8e8ef",
        outline="")
    
    img3 = PhotoImage(file=f'graphs\img_c.png')
    b3 = Button(
        main_frame,
        image=img3,
        borderwidth=0,
        highlightthickness=0,
        command=btn_clicked,
        relief="flat")
    
    b3.place(
        x=36, y=321,
        width=450,
        height=200)
    
    canvas.create_rectangle(
        515, 321, 515 + 450, 321 + 200,
        fill="#e8e8ef",
        outline="")
    
    img4 = PhotoImage(file=f'graphs\img_d.png')
    b4 = Button(
        main_frame,
        image=img4,
        borderwidth=0,
        highlightthickness=0,
        command=btn_clicked,
        relief="flat")
    
    b4.place(
        x=515, y=321,
        width=450,
        height=200)


    # Button - back to results
    img0 = PhotoImage(file = f"back.png")
    b0 = Button(
        main_frame,
        image = img0,
        borderwidth = 0,
        highlightthickness = 0,
        command = lambda : results_page(main_frame, username, recipt, user_id),
        relief = "flat")

    b0.place(
        x = 919, y = 533,
        width = 33,
        height = 30)
    
    window.protocol("WM_DELETE_WINDOW", lambda:on_closing(user_id, recipt))
    window.mainloop()
    
def overall_results_page(data, main_frame, username, recipt, user_id):
    
    
    graphs_in_time(data)
    
    img = cv2.imread('graphs\img_a.png')
    img_a = cv2.resize(img,(450,200))
    cv2.imwrite('img_a.png', img_a)

    
    img = cv2.imread('graphs\img_b.png')
    img_b = cv2.resize(img,(450,200))
    cv2.imwrite('img_b.png', img_b)

    
    img = cv2.imread('graphs\img_c.png')
    img_c = cv2.resize(img,(450,200))
    cv2.imwrite('img_c.png', img_c)

    
    img = cv2.imread('graphs\img_d.png')
    img_d = cv2.resize(img,(450,200))
    cv2.imwrite('img_d.png', img_d)

    # Clear frame:
    main_frame.destroy()
    main_frame = Frame(window)
    main_frame.pack(fill=BOTH, expand=1)

    canvas = Canvas(
        main_frame,
        bg="#ffffff",
        height=600,
        width=1000,
        bd=0,
        highlightthickness=0,
        relief="ridge")
    canvas.place(x=0, y=0)

    background_img = PhotoImage(file=f"background_dayli.png")
    background = canvas.create_image(
        500.0, 300.0,
        image=background_img)


    # Title
    canvas.create_text(
        500.0, 44.5,
        text = "תוצאות כלליות",
        fill = "#509cf6",
        font = ("IstokWeb-Regular", int(30.0)))


    # Charts:
    canvas.create_rectangle(
        515, 87, 515 + 450, 87 + 200,
        fill="#e8e8ef",
        outline="")
    
    img1 = PhotoImage(file=f'img_a.png')
    b1 = Button(
        main_frame,
        image=img1,
        borderwidth=0,
        highlightthickness=0,
        command=btn_clicked,
        relief="flat")

    b1.place(
        x=515, y=87,
        width=450,
        height=200)
    
    canvas.create_rectangle(
        36, 87, 36 + 450, 87 + 200,
        fill="#e8e8ef",
        outline="")
    
    img2 = PhotoImage(file=f'img_b.png')
    b2 = Button(
        main_frame,
        image=img2,
        borderwidth=0,
        highlightthickness=0,
        command=btn_clicked,
        relief="flat")
    
    b2.place(
        x=36, y=87,
        width=450,
        height=200)

    canvas.create_rectangle(
        36, 321, 36 + 450, 321 + 200,
        fill="#e8e8ef",
        outline="")
    
    img3 = PhotoImage(file=f'img_c.png')
    b3 = Button(
        main_frame,
        image=img3,
        borderwidth=0,
        highlightthickness=0,
        command=btn_clicked,
        relief="flat")
    
    b3.place(
        x=36, y=321,
        width=450,
        height=200)
    
    canvas.create_rectangle(
        515, 321, 515 + 450, 321 + 200,
        fill="#e8e8ef",
        outline="")
    
    img4 = PhotoImage(file=f'img_d.png')
    b4 = Button(
        main_frame,
        image=img4,
        borderwidth=0,
        highlightthickness=0,
        command=btn_clicked,
        relief="flat")
    
    b4.place(
        x=515, y=321,
        width=450,
        height=200)


    # Button - back to results
    img0 = PhotoImage(file = f"back.png")
    b0 = Button(
        main_frame,
        image = img0,
        borderwidth = 0,
        highlightthickness = 0,
        command = lambda : progress_page(main_frame, username, recipt, user_id),
        relief = "flat")

    b0.place(
        x = 919, y = 533,
        width = 33,
        height = 30)
    
    window.protocol("WM_DELETE_WINDOW", lambda:on_closing(user_id, recipt))
    window.mainloop()

def progress_page (main_frame, username, recipt, user_id):

    # Clear frame:
    main_frame.destroy()
    main_frame = Frame(window)
    main_frame.pack(fill=BOTH, expand=1)

    open_mouth, kiss, smile, down, up, left, right, aa, eh, wuw, ow, iy = [], [] ,[] ,[], [], [], [], [], [], [], [], []

    data = pd.read_csv(str(recipt)+'.csv')
    data['Dates'] = pd.to_datetime(data['date']).dt.date
    moves = data['move'].unique()
    data = data.set_index('move')
    data = data.drop(columns=['date'])

    d ={}
    for move in moves:
        d["{0}".format(move)] = data.loc[move]
    
    canvas = Canvas(
        main_frame,
        bg="#ffffff",
        height=600,
        width=1000,
        bd=0,
        highlightthickness=0,
        relief="ridge")
    canvas.place(x=0, y=0)

    background_img = PhotoImage(file=f"background_progress.png")
    background = canvas.create_image(
        500.0, 300.0,
        image=background_img)

    # Title
    canvas.create_text(
        501.5, 57.0,
        text = "תוצאות אימונים",
        fill = "#509cf6",
        font = ("IstokWeb-Regular", int(40.0)))

    # Button - back to results_page
    img0 = PhotoImage(file=f"back.png")
    b0 = Button(
        main_frame,
        image=img0,
        borderwidth=0,
        highlightthickness=0,
        command=lambda : results_page(main_frame,username, recipt, user_id),
        relief="flat")

    b0.place(
        x=805, y=531,
        width=33,
        height=30)

    if 'down' in d:
        down = d['down']

    # Button - Tongue down
    img1 = PhotoImage(file=f"img1_progress.png")
    b1 = Button(
        main_frame,
        image=img1,
        borderwidth=0,
        highlightthickness=0,
        command= lambda: overall_results_page(down, main_frame ,username, recipt, user_id),
        relief="flat")

    b1.place(
        x=183, y=113,
        width=110,
        height=107)

    if 'up' in d:
        up = d['up']
        
    # Button - Tongue up
    img2 = PhotoImage(file=f"img2_progress.png")
    b2 = Button(
        main_frame,
        image=img2,
        borderwidth=0,
        highlightthickness=0,
        command= lambda: overall_results_page(up, main_frame, username, recipt, user_id),
        relief="flat")

    b2.place(
        x=359, y=113,
        width=110,
        height=107)

    if 'right' in d:
        right = d['right']

    # Button - Tongue right
    img3 = PhotoImage(file=f"img3_progress.png")
    b3 = Button(
        main_frame,
        image=img3,
        borderwidth=0,
        highlightthickness=0,
        command= lambda: overall_results_page(right, main_frame, username, recipt, user_id),
        relief="flat")

    b3.place(
        x=535, y=113,
        width=110,
        height=107)

    if 'left' in d:
        left = d['left']

    # Button - Tongue left
    img4 = PhotoImage(file=f"img4_progress.png")
    b4 = Button(
        main_frame,
        image=img4,
        borderwidth=0,
        highlightthickness=0,
        command= lambda: overall_results_page(left, main_frame, username, recipt, user_id),
        relief="flat")

    b4.place(
        x=712, y=113,
        width=110,
        height=107)

    if 'open_mouth' in d:
        open_mouth = d['open_mouth']

    # Button - Open mouth
    img5 = PhotoImage(file=f"img5_progress.png")
    b5 = Button(
        main_frame,
        image=img5,
        borderwidth=0,
        highlightthickness=0,
        command= lambda: overall_results_page(open_mouth, main_frame, username, recipt, user_id),
        relief="flat")

    b5.place(
        x=258, y=252,
        width=110,
        height=107)

    if 'smile' in d:
        smile = d['smile']


    # Button - Smile
    img6 = PhotoImage(file=f"img6_progress.png")
    b6 = Button(
        main_frame,
        image=img6,
        borderwidth=0,
        highlightthickness=0,
        command= lambda: overall_results_page(smile, main_frame, username, recipt, user_id),
        relief="flat")

    b6.place(
        x=447, y=254,
        width=110,
        height=107)

    if 'kiss' in d:
        kiss = d['kiss']


    # Button - Kiss
    img7 = PhotoImage(file=f"img7_progress.png")
    b7 = Button(
        main_frame,
        image=img7,
        borderwidth=0,
        highlightthickness=0,
        command= lambda: overall_results_page(kiss, main_frame, username, recipt, user_id),
        relief="flat")

    b7.place(
        x=637, y=254,
        width=110,
        height=107)

    if 'aa' in d:
        aa = d['aa']

    # Button - A
    img8 = PhotoImage(file=f"img8_progress.png")
    b8 = Button(
        main_frame,
        image=img8,
        borderwidth=0,
        highlightthickness=0,
        command= lambda: overall_results_page(aa, main_frame, username, recipt, user_id),
        relief="flat")

    b8.place(
        x=183, y=415,
        width=90,
        height=65)

    if 'eh' in d:
        eh = 'eh'


    # Button - E
    img9 = PhotoImage(file=f"img9_progress.png")
    b9 = Button(
        main_frame,
        image=img9,
        borderwidth=0,
        highlightthickness=0,
        command= lambda: overall_results_page(eh, main_frame, username, recipt, user_id),
        relief="flat")

    b9.place(
        x=309, y=415,
        width=90,
        height=65)


    if 'wuw' in d:
        wuw = d['wuw']


    # Button - U
    img10 = PhotoImage(file=f"img10_progress.png")
    b10 = Button(
        main_frame,
        image=img10,
        borderwidth=0,
        highlightthickness=0,
        command= lambda: overall_results_page(wuw, main_frame, username, recipt, user_id),
        relief="flat")

    b10.place(
        x=447, y=415,
        width=90,
        height=65)

    if 'ow' in d:
        ow = d['ow']

    # Button - O
    img11 = PhotoImage(file=f"img11_progress.png")
    b11 = Button(
        main_frame,
        image=img11,
        borderwidth=0,
        highlightthickness=0,
        command= lambda: overall_results_page(ow, main_frame, username, recipt, user_id),
        relief="flat")

    b11.place(
        x=583, y=415,
        width=90,
        height=65)

    if 'iy' in d:
        iy = d['iy']


    # Button - I
    img12 = PhotoImage(file=f"img12_progress.png")
    b12 = Button(
        main_frame,
        image=img12,
        borderwidth=0,
        highlightthickness=0,
        command= lambda: overall_results_page(iy, main_frame, username, recipt, user_id),
        relief="flat")

    b12.place(
        x=721, y=415,
        width=90,
        height=65)
    
    window.protocol("WM_DELETE_WINDOW", lambda:on_closing(user_id, recipt))
    window.mainloop()


# main:

# Data Base
# x = pd.DataFrame(columns=['name','receipt','password'])
# x.to_csv("thera_speech_database.csv",index=False)
# x=pd.read_csv('thera_speech_database.csv')

file_id_database_dir = '1eKiyCyDUisO69tR54Bg5cY9PSMK3NMtr'
#file_id_users_dir = "1ZMUU75z9DBG1g4Wh84pvRNPa6xBT73C3"
thera_id = '1ZkgWvHaTlCLQ3cPn_NwdUqqv4XuvgiEt'

# obj = DriveAPI()
# obj.FileDownload(thera_id, 'thera_speech_database.csv')

mask_type=None
mask_tune=[]
np.save("mask/mask_type.npy",mask_type)
np.save("mask/mask_tune.npy",mask_tune)

# Window setings:

window = Tk()  # window on the system
window.geometry("1000x600")  # window shape
window.configure(bg="#ffffff")  # Query or set the default value of the specified option(s) in style.
window.title("Tera-Speech")  # window title
window.iconbitmap('person.ico')  # window iconframe=Frame(window)

# Create A Main Frame
main_frame = Frame(window)
main_frame.pack(fill=BOTH, expand=1)

home_page(main_frame)

# means, std = get_var_image()





