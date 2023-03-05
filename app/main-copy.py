# import requirements needed
from flask import Flask, redirect, url_for, request
from utils import get_base_url
from model import Person
from flask import Flask, render_template


# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12341
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

@app.route(f'{base_url}', methods = ['POST','GET'])
def home():
#     Return prediction info from model: 1 or 0 (Maybe percentage of having heart disease or something)
    if request.method == 'POST':
        age = request.form['age']
        sex = request.form['sex']
        patient = Person(sex, age)
        result = patient.predict()
        ans = ''
        if result == 1:
            ans = "You are at risk of having heart diease"
        else:
            ans = "You do not have to worry about heart disease"
        return render_template('index.html', var = ans)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'cocalc21.ai-camp.dev/'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)

    
# Get and post methods: we're figuring out how we handle info and spit it out
# Don't save the data
# Inputting
#     Neg correlation(age,thalach)
#     pos correlation (female, if age == mid 50s, slope)
