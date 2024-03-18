import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={"GREScore":200,  "TOEFLScore":95,  "UniversityRating":3,  "SOP":4.5,  "LOR":4.5,  "CGPA":5.5})

print(r.json())

# [200, 95, 3, 4.5, 4.5, 5.5]