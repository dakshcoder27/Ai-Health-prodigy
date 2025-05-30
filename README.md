## Project Name 
__Ai Health Prodigy An AI based Health-CheckUp Web Tool__


## Motivation

- Whenever a patient visits a hospital it takes a significant amount of time before the updated health reports of the patient arrives making it difficult for the proper detection and hence decision making for the health official. 
- Meanwhile, others experience ailments but are unable to afford a visit to the doctor due to a lack of or poor health care. Further, if the patient recognizes his/her symptoms, and if somehow we can tell him what is the disease he is likely to be affected with then he/she can take precautions accordingly at home only.
- Since, everyone should have easy access to great health care there is a need to connect patients virtually with doctors.
- So, our project aims to not only effectively connect doctors and patients virtually but incase if a patient recognises the symptoms, then he/she can know what disease he/she is likely to be infected with and what precautionary measures can be taken with the help of Artificial Intelligence

## Objective

The application mainly consists of three features:

- First, we have designed a computer-aided diagnosis system (or disease prediction system) where users can get to know whether they are infected with a particular disease or not using machine/deep learning. For this, they are required to enter their medical details on the form or upload MRI image.
- Secondly, there is a feature to enter the symptoms (either simply type the symptoms or record the audio in browser) they are experiencing and the patients will get to know what possible diseases they might have along with the precautions that they must take.
- Third feature is the doctor appointment system wherein patients can not only search doctors based on region or specialization, but also connect virtually with the doctors around the globe.

## Requirements:
- Frontend: Html5, CSS, JavaScript, Jquery, Bootstrap
- Backend: Django
- Database: SQLite 
- Machine/Deep Learning Frameworks: Scikit-Learn, Tensorflow/PyTorch
- Dataset platform: Kaggle
- Browser: Any web browser like Chrome, Firefox for running the web application


## Getting Started

**Step 1. Clone the repository into a new folder and then switch to code directory**

```
git clone https://github.com/gautamgc17/Health-CheckUp.git
cd Health-CheckUp
```

**Step 2. Create a Virtual Environment to install dependencies.**

```
pip install virtualenv
```

Create a new Virtual Environment for the project and activate the environment to install the libraries.

```
virtualenv env
env\Scripts\activate
```

Once the virtual environment is activated, the name of your virtual environment will appear on left side of terminal.

Next, we need to install the project dependencies in this virtual environment, which are listed in `requirements.txt`.

```
pip install -r requirements.txt
```

**Step3 . Download the trained models and include them in the models folder of the root directory.**

The trained deep learning models can be downloaded from [here](https://drive.google.com/drive/folders/1_A7VgM08sQ6Pgzb7ohmTb17EyRASAOCb).

**Step 4. Set up ffmpeg for speech to text conversion**

1. Go to: https://ffmpeg.org/download.html download the ffmpeg full release build zip file
2. Extract the zip file locate the ffmpeg.exe and copy the path and add it to enviroment variables

**Step 5. Update environment variables.**

To run the project, you need to configure the application to run locally. This will require updating a set of environment variables specific to your environment.

In the same directory, create a local environment file, named - `.env`.

_Now simply duplicate the variables in **.env.sample** file and just insert your credentials into local environment file - `.env`._

**Step 6. Run Django Project.**

- Make migrations to create/apply changes to the models into the database schema.

```
python manage.py makemigrations
python manage.py migrate
```

- Create a superuser for django admin panel.

```
python manage.py createsuperuser
```

- Insert some dummy doctor data into the sqlite database and then finally run the server code.

```
python manage.py runserver
```

## Website Screenshots

## Front Page

![front](website-screenshots/FrontPage.png)

## Covid Prediction

![Covid Prediction](website-screenshots/Covid.png)

## Liver Form

![Liver Form](website-screenshots/Liver.png)

## Liver Form Result

![Liver Form Result](website-screenshots/LiverRes.png)

## Know Your Disease

![Know Your Disease](website-screenshots/KYDpage.png)

## Know Your Disease Symptom

![Know Your Disease Symptom](website-screenshots/KYDSym.png)

## Know Your Disease Audio

![Know Your Disease Audio](website-screenshots/KYDAudio.png)

## Know Your Disease Symptom

![Know Your Disease Symptom](website-screenshots/KYDAudio.png)

## Registeration Page

![Registeration](website-screenshots/Register.png)

## Login Page

![Login Page](website-screenshots/Login.png)

## Display of Doctors Database

![Top 5 Doctors add to Database](website-screenshots/top5doct.png)

## Filtered Doctors list based on region/specialization

![Filtered Doctor](website-screenshots/Filtereddoc.png)

## Time Slots for Booking

![Booking Time Slot](website-screenshots/Booking.png)


