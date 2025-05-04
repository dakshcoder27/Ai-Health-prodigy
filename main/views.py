from django.shortcuts import render , redirect
from django.http import HttpResponse , HttpResponseRedirect , JsonResponse , FileResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from models.classifiers import ( predict_liverD, predict_alzheimer, 
 predict_brain, localizeTumor, predict_disease)
from models.transcribe import transcribe_audio
import os
import re
import base64
import numpy as np
from PIL import Image
import subprocess


# Create your views here
def front(request):
    return render(request , 'front/index.html')

def bmiCalc(request):
    return render(request , 'bmi/bmi-calculator.html')



def liver(request):
    return render(request , 'liver/index.html')

def liverPred(request):
    if request.method == 'POST':
        features = list(request.POST.dict().values())[1:]
        out_features = request.POST.dict()
        print(out_features)
        pred = predict_liverD(np.array([features]))
        context = {
            'pred':pred,
            'out_features':out_features
        }
        return render(request , 'liver/page2.html' , context)
    
    else:
        redirect('liver/')



def alzheimerPred(request):
    if request.method == 'POST'and request.FILES['image']:

        img = request.FILES['image']
        fss = FileSystemStorage()
        file = fss.save(img.name, img)
        file_url = fss.url(file)

        file_path = os.path.join(settings.MEDIA_ROOT, img.name)

        img = Image.open(img).convert("RGB").resize((224, 224))
        label , cam_path = predict_alzheimer(img , file_path , file_path.replace(".jpg", "") + "_camViz.jpg")

        with open(cam_path, "rb") as img2str:
            converted_string = base64.b64encode(img2str.read())

        context = {
            'file_url': cam_path,
            'label': label,
            'photo': str(converted_string)
        }
        return JsonResponse(context)
    
    else:
        return render(request , 'alzheimer/Alzheimer.html')



def brainPred(request):
    if request.method == 'POST' and request.FILES['image']:

        img = request.FILES['image']
        fss = FileSystemStorage()
        file = fss.save(img.name, img)
        file_path = os.path.join(settings.MEDIA_ROOT, img.name)

        img = Image.open(img).resize((224, 224))
        label = predict_brain(img)

        if label == "No Tumor":
            with open(file_path, "rb") as img2str:
                converted_string = base64.b64encode(img2str.read())
            file_url = file_path
        else:
            out_path = localizeTumor(file_path , file_path.replace(".jpg", "") + "_tumorLoc.jpg")
            with open(out_path, "rb") as img2str:
                converted_string = base64.b64encode(img2str.read())
            file_url = out_path

        context = {
            'file_url': file_url,
            'label': label,
            'photo': str(converted_string)
        }

        return JsonResponse(context)

    else:
        return render(request , 'brain/BrainTumor.html')




import pandas as pd

# Load valid symptom list once
training = pd.read_csv('models/training.csv')
valid_symptoms = [col.strip().lower() for col in training.columns[:-1]]

import av
import numpy as np
from scipy.io.wavfile import write
import mimetypes


def convert_to_wav(input_path):
    container = av.open(input_path)
    stream = next(s for s in container.streams if s.type == 'audio')

    samples = []
    for frame in container.decode(stream):
        samples.append(frame.to_ndarray())

    audio = np.concatenate(samples)
    write(input_path.replace('.webm', '.wav'), stream.rate, audio)
    return input_path.replace('.webm', '.wav')

def symptomsDis(request):
    if request.method == 'POST':
        # 🎙 Audio-based input
        if 'audio' in request.FILES:
            audio_data = request.FILES["audio"]
            fss = FileSystemStorage()
            file_name = audio_data.name + ".webm"

            file = fss.save(file_name, audio_data)
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)


            # file_path = convert_to_wav(file_path)
            mime_type, _ = mimetypes.guess_type(file_path)
            print(f"[DEBUG] Saved file path: {file_path}")
            print(f"[DEBUG] MIME type: {mime_type}")
            text = transcribe_audio(file_path)
            print(f"[DEBUG] Transcribed text: {text}")

            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            symptoms = [word for word in words if word in valid_symptoms]

            print(f"[SYMPTOMS EXTRACTED FROM AUDIO] {symptoms}")

            if not symptoms:
                return JsonResponse({
                    "error": "No valid symptoms recognized from audio.",
                    "recognized_text": text
                })

            advice, output = predict_disease(symptoms)
            return JsonResponse({
                "advice": advice,
                "output": output,
                "recognized_symptoms": symptoms
            })

        # ✍️ Text-based input
        values = list(request.POST.dict().values())
        user_symptoms = [val.strip().lower() for val in values[:-1]]
        days = int(values[-1]) if values[-1].isdigit() else 5

        advice, output = predict_disease(user_symptoms, days)
        return JsonResponse({
            "advice": advice,
            "output": output,
            "recognized_symptoms": user_symptoms
        })

    return render(request, 'soundRecorder/recorder.html')

