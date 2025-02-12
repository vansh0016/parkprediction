from flask import Flask, render_template, request
import pickle
import xgboost as xgb

app = Flask(__name__)

@app.route('/',methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def index():

    if request.method == 'POST':
            mdvp_fo=float(request.form['mdvp_fo'])
            mdvp_fhi=float(request.form['mdvp_fhi'])
            mdvp_flo=float(request.form['mdvp_flo'])
            mdvp_jitper=float(request.form['mdvp_jitper'])
            mdvp_jitabs=float(request.form['mdvp_jitabs'])
            mdvp_rap=float(request.form['mdvp_rap'])
            mdvp_ppq=float(request.form['mdvp_ppq'])
            jitter_ddp=float(request.form['jitter_ddp'])
            mdvp_shim=float(request.form['mdvp_shim'])
            mdvp_shim_db=float(request.form['mdvp_shim_db'])
            shimm_apq3=float(request.form['shimm_apq3'])
            shimm_apq5=float(request.form['shimm_apq5'])
            mdvp_apq=float(request.form['mdvp_apq'])
            shimm_dda=float(request.form['shimm_dda'])
            nhr=float(request.form['nhr'])
            hnr=float(request.form['hnr'])
            rpde=float(request.form['rpde'])
            dfa=float(request.form['dfa'])
            spread1=float(request.form['spread1'])
            spread2=float(request.form['spread2'])
            d2=float(request.form['d2'])
            ppe=float(request.form['ppe'])

            sc=pickle.Unpickler(open("scaler.pickle", "rb"))
            scaler=sc.load()

            print(scaler)

            loaded_model = xgb.Booster()
            loaded_model.load_model("model.json")

            x_val=[
            mdvp_fo,
            mdvp_fhi,
            mdvp_flo,
            mdvp_jitper,
            mdvp_jitabs,
            mdvp_rap,
            mdvp_ppq,
            jitter_ddp,
            mdvp_shim,
            mdvp_shim_db,
            shimm_apq3,
            shimm_apq5,
            mdvp_apq,
            shimm_dda,
            nhr,
            hnr,
            rpde,
            dfa,
            spread1,
            spread2,
            d2,
            ppe,
            ]

            dm=xgb.DMatrix(scaler.transform([x_val]))

            prediction=loaded_model.predict(dm)
            print('prediction is', prediction[0])
            if round(prediction[0]) == 1:
                pred = "You have Parkinson's Disease. Please consult a specialist."
            else:
                pred = "You are a Healthy Person."
            return render_template('results.html',prediction=pred)
    else:
        return render_template('index.html')

if __name__ == "__main__":
	app.run(debug=True)