import base64
print(base64.b64encode(open("models/disease_model.keras", "rb").read()).decode())
