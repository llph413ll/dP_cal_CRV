from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import datetime

app = Flask(__name__)

# Load the data and train the model
data = pd.read_csv('CRV_dP_parametric.csv')
X = data[['fluid_density[lb/ft^3]', 'fluid_viscosity[centipoise]', 'velocity[ft/s]']]
y = data['dP[psi]']
rf_model = RandomForestRegressor(n_estimators=100, random_state=10)
rf_model.fit(X, y)
current = datetime.datetime.now().strftime('%m-%d-%Y')

density_conversions = {'lbft3': 1, 'kgm3': 1/16.0185, 'gcm3': 62.427, 'gl':0.062427}  # Example: 1 lb/ft³ to 16.0185 kg/m³
viscosity_conversions = {'cp': 1, 'mpas': 1, 'pas':1000, 'nsm2':1000}  # 1 centipoise to 1 mPa·s (same unit essentially)
velocity_conversions = {'fts': 1, 'ms': 3.28084, 'mh':1.46666, 'ins':0.0833}  # 1 m/s to 3.28084 ft/s


@app.route('/')
def home():
    return render_template('index.html', time=current)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values and units from form
        density = float(request.form['density'])
        density_unit = request.form['density_unit']
        viscosity = float(request.form['viscosity'])
        viscosity_unit = request.form['viscosity_unit']
        velocity = float(request.form['velocity'])
        velocity_unit = request.form['velocity_unit']

        # Convert all inputs to standard units defined in your model
        density *= density_conversions[density_unit]
        viscosity *= viscosity_conversions[viscosity_unit]
        velocity *= velocity_conversions[velocity_unit]
        valid_ranges = {
            'fluid_density[lb/ft^3]': (49.92, 81.12),
            'fluid_viscosity[centipoise]': (0.8899, 50),
            'velocity[ft/s]': (1, 12)
        }
        features = [density, viscosity, velocity]
        feature_names = ['fluid_density[lb/ft^3]', 'fluid_viscosity[centipoise]', 'velocity[ft/s]']

        for feature_name, value in zip(feature_names, features):
            min_val, max_val = valid_ranges[feature_name]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{feature_name} entered as {value} exceeds the allowable range."
                                f"</br>Enter a value between the minimum of {min_val} and the maximum of {max_val}.")

        prediction = rf_model.predict([features])[0]
        return render_template('index.html', prediction_text=f"Pressure drop: {prediction:.2f} [psi]",time=current)
    except ValueError as e:
        return render_template('index.html', error_message=str(e),time=current)

if __name__ == "__main__":
    app.run(debug=True)

