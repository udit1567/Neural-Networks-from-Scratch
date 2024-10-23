import torch
import torch.nn as nn

class NeuralNetworkClassificationModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetworkClassificationModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, 50)
        self.hidden_layer1 = nn.Linear(50, 60)
        self.output_layer = nn.Linear(60, output_dim)
        self.relu = nn.ReLU()
        self.tan = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.input_layer(x))
        out = self.relu(self.hidden_layer1(out))
        out = self.output_layer(out)
        return out

input_dim  = 6
output_dim = 3
model = NeuralNetworkClassificationModel(input_dim,output_dim)


model.load_state_dict(torch.load('penguin.pth',weights_only=True))

model.eval()
from flask import Flask, render_template, request

app = Flask(__name__)

import torch

import torch

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Capture the form data
        island = request.form.get('island')
        bill_length_mm = request.form.get('bill_length_mm')
        bill_depth_mm = request.form.get('bill_depth_mm')
        flipper_length_mm = request.form.get('flipper_length_mm')
        body_mass_g = request.form.get('body_mass_g')
        sex = request.form.get('sex')


        island_mapping = {'Torgersen': 0, 'Biscoe': 1, 'Dream': 2}
        sex_mapping = {'Male': 0, 'Female': 1}

        island_mapped = island_mapping.get(island)
        sex_mapped = sex_mapping.get(sex)


        bill_length_mm = float(bill_length_mm)
        bill_depth_mm = float(bill_depth_mm)
        flipper_length_mm = float(flipper_length_mm)
        body_mass_g = float(body_mass_g)


        a = torch.tensor([island_mapped, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex_mapped])

        b = torch.argmax(model(a))
        print(b)


        species_mapping = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
        species = species_mapping.get(b.item())
        print(species)

        return f"Respective species is {species}"

    return render_template('form.html')



if __name__ == '__main__':
    app.run(debug=True)