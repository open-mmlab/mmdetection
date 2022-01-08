import requests

# https://your-heroku-app-name.herokuapp.com/predict  on web server
# http://localhost:5000/predict  in localhost
resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('../test_imgs/astro[hippo]_D1-1_Vessel-361_2020-09-14_13h00m00s_Ph_1.png', 'rb')})

print(resp.text)