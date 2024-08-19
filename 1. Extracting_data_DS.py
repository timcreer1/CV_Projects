import numpy as np
from scipy.signal import resample
import os
from pathlib import Path
import pandas as pd
import zipfile
import time
from tqdm import tqdm

# Function to check if all elements in a list are equal
def equality(lst):
    return len(set(lst)) == 1

# Define the path to the spectra data
spectra_path = Path('/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/Data/Rapid_age-grading_and_species_identification_of_natural_mosquitoes')

# Collect all .dpt and .mzz files in the specified directory
dptfiles = [Path(root) / file for root, _, files in os.walk(spectra_path) for file in files if file.endswith(".dpt")]
mzzfiles = [Path(root) / file for root, _, files in os.walk(spectra_path) for file in files if file.endswith(".mzz")]
spectra_names = mzzfiles if mzzfiles else dptfiles
mzzq = bool(mzzfiles)

# Check the consistency of file naming sections
naimeision = [len(os.path.basename(i).split("-")) for i in spectra_names]
if not equality(naimeision):
    lf = list(set(naimeision))
    nf = [naimeision.count(j) for j in lf]
    jf = [os.path.basename(spectra_names[k]) for k, l in enumerate(naimeision) if l == lf[nf.index(min(nf))]]
    print("Attention!!!!\nNot all files have the same number of sections. These are probably the files that are misnamed:\n")
    for j in jf:
        print(f"     {j}")
else:
    print("Everything seems all right. You may continue.")

# Further consistency checks and preparation for data extraction
if equality(naimeision):
    tmp2 = naimeision[0]
    tembo = [[] for _ in range(tmp2)]
    for i in spectra_names:
        tmp = os.path.basename(i).split("-")
        tmp[-1] = tmp[-1].split(".")[0].split(" ")[0]
        for k in range(min(len(tmp), tmp2)):
            tembo[k].append(tmp[k].upper())
    hakuna = [list(set(tembo[i])) for i in range(tmp2)]
    matata = ["ID" if len(hakuna[i]) > 0.4 * len(spectra_names) else "Dat" if len(hakuna[i][0]) == 6 and (hakuna[i][0][0] in "12") else "Cat" for i in range(tmp2)]
    jf = [os.path.basename(spectra_names[m]) for j, l in enumerate(matata) if l == "Dat" for m, n in enumerate(tembo[j]) if len(n) != 6]
    if jf:
        print("Attention!!!!\nThere are files with wrong date format. These are probably the files that are misnamed:\n")
        for j in jf:
            print(f"     {j}")
else:
    print("I told you this was not going to run unless you solve the problem with the names")

# Initialize data structures for spectra and misnamed files
matrix, wrong_named = [], []
mbuni, kifaru = [m for m, n in enumerate(matata) if n != "Dat"], [m for m, n in enumerate(matata) if n == "Dat"]

# Process each spectrum file
for i in tqdm(spectra_names[:52000]):
    tmp = os.path.basename(i).split("-")
    tmp[-1] = tmp[-1].split(".")[0].split(" ")[0]
    for k in range(len(tmp)):
        tmp[k] = tmp[k].upper()

    # Load the spectrum data from the file
    if not mzzq:
        with open(i, 'rb') as tmp_file:
            spectrum = np.genfromtxt((line.replace(b'\t', b',') for line in tmp_file), delimiter=',')
        start, end, ls = spectrum[0, 0], spectrum[-1, 0], len(spectrum)
        spectrum = np.transpose(spectrum)[1]
    else:
        with zipfile.ZipFile(i) as myzip:
            with myzip.open(myzip.namelist()[0]) as myfile:
                spectrum = np.genfromtxt(myfile, delimiter=',')
        start, end, ls = spectrum[0], spectrum[1], int(spectrum[2])
        spectrum = spectrum[3:]

    # Prepare the data entry for the matrix
    fisi = [[start, end, ls], spectrum] + [tmp[j] for j in mbuni]
    if len(kifaru) == 2:
        try:
            colday = time.mktime(time.strptime(tmp[kifaru[0]], "%y%m%d"))
            mesday = time.mktime(time.strptime(tmp[kifaru[1]], "%y%m%d"))
            stime = round(abs((mesday - colday) / (3600 * 24)))
            fisi.append(stime)
        except:
            wrong_named.append(i)
            continue
    matrix.append(fisi)

# Downsample the spectra for further analysis
downsampled_matrix = [[row[0], ','.join(map(str, resample(row[1], 1000)))] + row[2:] for row in matrix]

# Create a DataFrame from the downsampled data
df_columns = ['Wavenumber Range', 'Spectrum'] + [f'Attribute_{i}' for i in range(1, len(downsampled_matrix[0]) - 1)]
df = pd.DataFrame(downsampled_matrix, columns=df_columns)

# Save the DataFrame to a CSV file
output_file_path = '/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/Data/matrix_downsampled.csv'
df.to_csv(output_file_path, index=False)

# Print the DataFrame to verify the results
print(df)
