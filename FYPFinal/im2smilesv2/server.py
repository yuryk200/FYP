import cv2
import json
import requests
import numpy as np
from collections import Counter
from flask import Flask, request, jsonify
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import inchi
from rdkit.Chem import rdMolDescriptors
import openai
import re

from model.img2seq import Img2SeqModel
from model.utils.general import Config
from model.utils.text import Vocab

app = Flask(__name__)


@app.route('/interactive_shell', methods=['POST'])

def interactive_shell():
    
    file = request.files['image'].read()
    
    # Retriving image 
    tmp = np.frombuffer(file, dtype=np.uint8)
    
    # Scanning Image retrived from app
    img = cv2.imdecode(tmp, flags=cv2.IMREAD_COLOR)

    dir_output = "Z:/FYP/im2smilesv2/results/clean-RDKit-500K/"
    config_vocab = Config(dir_output + "vocab.json")
    config_model = Config(dir_output + "model.json")
    vocab = Vocab(config_vocab)

    # loading model
    model = Img2SeqModel(config_model, dir_output, vocab)

    # running model
    model.build_pred()
    model.restore_session(dir_output + "model.weights/")

    # chaning image color from RBG to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # resizing the image
    img = cv2.resize(img, (256, 256))
    
    # changing NumPy array
    # adding new axis to make into 3 dimensonal array for grayscale
    img = img[:, :, np.newaxis].astype(np.uint8)

    # sending model data to a list of hypothesis
    hyps = model.predict(img)

    # selecting most likely hypothesis
    smiles = hyps[0]

    # converting to moles
    mol = Chem.MolFromSmiles(smiles)
    
    try:
        # calculating moles formula
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    except Exception as e:
        print("Error in structure, placement structure will be used instead")
        formula = "Error: {}".format(str(e))
        smiles = "C=C1C=CCCC1=CC1CCC1"
        mol = Chem.MolFromSmiles(smiles)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)

    # adding hydrogen to the atom
    mol = Chem.AddHs(mol)

    # generate 3D atoms
    AllChem.EmbedMolecule(mol)

    AllChem.MMFFOptimizeMolecule(mol)

    # Write the SDF string to a file
    with open("output.sdf", "w") as f:
        f.write(Chem.MolToV3KMolBlock(mol))

    inchi_str = inchi.MolToInchi(mol)
    print(inchi_str)

    #Using chatGPT to translate to Inchi stringS
    openai.api_key = "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"
    prompt = f"Find using OPSIN what is the compound name for the following inchi string: {inchi_str}?"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract IUPAC name from the response
    iupac_name = response.choices[0].text
    iupac_name = re.sub('(\n|\r)', '', iupac_name)  # Remove line breaks
    iupac_name = iupac_name.replace(prompt, '').strip()
    print(iupac_name)

    print("chemical formula: " + formula)
    #json_string = json.dumps({"positions": coords})

    sdf_str = Chem.MolToV3KMolBlock(mol)

    return {"smiles": smiles, "formula": formula, "iupac_name": iupac_name, "sdf": sdf_str}




if __name__ == "__main__":
    app.run(host="0.0.0.0", use_reloader=False)
