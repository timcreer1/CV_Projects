import pickle
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# to manage the files
import os
# to deal with file paths on Windows, Mac and Linux
from pathlib import Path
# to manage the data textfiles
import csv
# to decompress the mzz files
import zipfile
import zlib
import time
# to know the progress in the slow parts
from tqdm import tqdm

# a quick algorithm to check if all the names have the same number of sections
def equality(listina):
    listina = iter(listina)
    try:
        uno = next(listina)
    except StopIteration:
        return True
    return all(uno == rest for rest in listina)

spectra_path = '/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/Data/Rapid_age-grading_and_species_identification_of_natural_mosquitoes'

# we include the path module that helps a lot with the / or \ problem
spectra_path = Path(spectra_path)
# find all the .mzz and .dpt files in the folder (and its subfolders)
dptfiles = []
mzzfiles = []
for root, dirs, files in os.walk(spectra_path):
    for file in files:
        if file.endswith(".dpt"):
             dptfiles.append(Path(root) / Path(file))
        elif file.endswith(".mzz"):
             mzzfiles.append(Path(root) / Path(file))
if len(mzzfiles) > 0:
    spectra_names = mzzfiles
    mzzq = True
else:
    spectra_names = dptfiles
    mzzq = False
naimeision = []
for i in spectra_names:
    # To obtain the info from the name of the file, first we split the name in the different sections:
    tmp = os.path.basename(i).split("-")
    tmp[-1] = tmp[-1].split(".")[0].split(" ")[0]
    tmp2 = len(tmp)
    naimeision.append(tmp2)
if equality(naimeision) == False:
    nf = []
    lf = list(set(naimeision))
    for j in lf:
        nf.append(naimeision.count(j))
    jf = []
    for k,l in enumerate(naimeision):
        if l == lf[nf.index(min(nf))]:
            jf.append(os.path.basename(spectra_names[k]))
    print("Attention!!!!")
    print("Not all files have the same number of sections. These are probably the files that are misnamed:")
    print("")
    for j in jf:
        print("     "+j)
else:
    print("Everything seems all right. You may continue.")

if equality(naimeision) == True:
    tembo = [[] for jijiji in range(tmp2)]
    for i in spectra_names:
        # To obtain the info from the name of the file, first we split the name in the different sections:
        tmp = os.path.basename(i).split("-")
        tmp[-1] = tmp[-1].split(".")[0].split(" ")[0]
        for k in range(len(tmp)):
            tmp[k] = tmp[k].upper()
        for k in range(tmp2):
            tembo[k].append(tmp[k])
    hakuna = [[] for jijiji in range(tmp2)]
    matata = []
    for i in range(len(tembo)):
        hakuna[i] = list(set(tembo[i]))

        if len(hakuna[i]) > 0.4 * len(spectra_names):
            matata.append("ID")
        else:
            if len(hakuna[i][0]) == 6 and (hakuna[i][0][0] == "1" or hakuna[i][0][0] == "2"):
                matata.append("Dat")
            else:
                matata.append("Cat")
    jf = []
    for j, l in enumerate(matata):
        if l == "Dat":
            lf = []
            for m, n in enumerate(tembo[j]):
                if len(n) != 6:
                    jf.append(os.path.basename(spectra_names[m]))
    if len(jf) > 0:
        print("Attention!!!!")
        print("There are files with wrond date format. These are probably the files that are misnamed:")
        print("")
        for j in jf:
            print("     " + j)
else:
    print("I told you this was not going to run unless you solve the problem with the names")

matrix = []
mbuni = [m for m, n in enumerate(matata) if n != "Dat"]
kifaru = [m for m, n in enumerate(matata) if n == "Dat"]
wrong_named = []
counter = 0  # Initialize a counter

# Now we load the spectra in a matrix
for i in tqdm(spectra_names):
    if counter >= 52000:  # Stop after loading 5000 items
        break

    # To obtain the info from the name of the file, first we split the name in the different sections:
    tmp = os.path.basename(i).split("-")
    tmp[-1] = tmp[-1].split(".")[0].split(" ")[0]
    for k in range(len(tmp)):
        tmp[k] = tmp[k].upper()

    # First the spectrum and its characteristics
    if not mzzq:
        with open(i, 'rb') as tmp:
            avmi = (line.replace(b'\t', b',') for line in tmp)
            spectrum = np.genfromtxt(avmi, delimiter=',')
        start = spectrum[0, 0]
        end = spectrum[-1, 0]
        ls = len(spectrum)
        spectrum = np.transpose(spectrum)[1]
    else:
        with zipfile.ZipFile(i) as myzip:
            tmpname = myzip.namelist()[0]
            with myzip.open(tmpname) as myfile:
                spectrum = np.genfromtxt(myfile, delimiter=',')
        start = spectrum[0]
        end = spectrum[1]
        ls = int(spectrum[2])
        spectrum = spectrum[3:]

    # And then we incorporate all the info to the matrix
    fisi = [[start, end, ls], spectrum] + [tmp[j] for j in mbuni]
    if len(kifaru) == 2:
        try:
            colday = time.mktime(time.strptime(tmp[kifaru[0]], "%y%m%d"))
        except:
            wrong_named.append(i)
            continue
        try:
            mesday = time.mktime(time.strptime(tmp[kifaru[1]], "%y%m%d"))
        except:
            wrong_named.append(i)
            continue
        stime = round(abs((mesday - colday) / (3600 * 24)))
        fisi.append(stime)

    matrix.append(fisi)
    counter += 1  # Increment the counter

# A list of the discarted spectra will be collected:
bad_spectra = []
for i in range(len(matrix)):
    # first we calculate the position of the points that comprise that section of the spectrum
    if matrix[i][0][1] < 600 and matrix[i][0][1] > 400:
        sta = int(round((((matrix[i][0][2] - 1) / (matrix[i][0][1] - matrix[i][0][0])) * (600 - matrix[i][0][0])) + 1)) - 1
        end = matrix[i][0][2]
    elif matrix[i][0][1] <= 400:
        sta = int(round((((matrix[i][0][2] - 1) / (matrix[i][0][1] - matrix[i][0][0])) * (600 - matrix[i][0][0])) + 1)) - 1
        end = int(round((((matrix[i][0][2] - 1) / (matrix[i][0][1] - matrix[i][0][0])) * (400 - matrix[i][0][0])) + 1)) - 1
    else:
        sta = 0 # if the spectrum doesn't reach 600 cm-1 we cannot prove if the spectrum has enough intensity
        raise Exception("The spectrum {} doesn't reach 600 cm-1".format(spectra_names[1]))
    # Now we check the intensity of the spectra in that region. If is not over 0.1 we discard the spectrum
    if np.average(matrix[i][1][sta:end]) < 0.11:
        bad_spectra.append("LI: " + str(spectra_names[i]))
        matrix[i] = None
if (bad_spectra) == 1:
    print("1 spectrum has been discarded because its low intensity")
else:
    print(str(len(bad_spectra)) + " spectra have been discarded because their low intensity")

bs = 0 # counter for the number of spectra discarderd
# we calculate the fences of the data set based in a value we can choose (in statistics 1.5 times
# the interquartile range is the inner fence and 3 times is the outer fence)
l = 2.5
# We look for the point at 1900 cm-1 and add it to the list of intensities
li = []
for i in range(len(matrix)):
    if matrix[i]: #to check if we have spectra
        # Now one would spect that the spectrum will reach 3900 so the program will not check it out.
        sta = int(round((((matrix[i][0][2] - 1) / (matrix[i][0][1] - matrix[i][0][0])) * (1900 - matrix[i][0][0])) + 1)) - 1
        li.append(matrix[i][1][sta])
q3, q1 = np.percentile(li, [75 ,25])
ir = q3 - q1
for i in range(len(matrix)):
    if matrix[i]: #to check if we have spectra
        sta = int(round((((matrix[i][0][2] - 1) / (matrix[i][0][1] - matrix[i][0][0])) * (1900 - matrix[i][0][0])) + 1)) - 1
        if matrix[i][1][sta] > (l * ir + q3) or matrix[i][1][sta] < (q1 - l * ir):
            bs +=1
            bad_spectra.append("SA: " + str(spectra_names[i]))
            matrix[i] = None
if (bs) == 1:
    print("1 spectrum has been discarded because it was distorted by the anvil")
else:
    print(str(bs) + " spectra have been discarded because they were distorted by the anvil")


bs = 0 # counter for the number of spectra discarderd
mycollection = []
# Now we define a function to calculate the R-squared coefficient of the fitting of our data to a polynomial
def rs_pf(x, y, degree):
    coeffs = np.polyfit(x, y, degree)
    # r-squared
    p = np.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results = ssreg / sstot

    return results

# Here take that the section of the data between 3900 and 3500 cm-1 and check if it fits well to a 5th degree polinomial
for i in range(len(matrix)):
    if matrix[i]: #to check if we have spectra
        # Now one would spect that the spectrum will reach 3900 so the program will not check it out.
        sta = int(round((((matrix[i][0][2] - 1) / (matrix[i][0][1] - matrix[i][0][0])) * (3900 - matrix[i][0][0])) + 1)) - 1
        end = int(round((((matrix[i][0][2] - 1) / (matrix[i][0][1] - matrix[i][0][0])) * (3500 - matrix[i][0][0])) + 1)) - 1
        # we take that data:
        yd = matrix[i][1][sta:end]
        xd = list(range(len(yd)))
        rs = rs_pf(xd,yd,5)
        # And now, if the fitting is bad, we discard the spectrum
        if rs < 0.96:
            bs +=1
            bad_spectra.append("AI: " + str(spectra_names[i]))
            matrix[i] = None
if (bs) == 1:
    print("1 spectrum has been discarded because has atmospheric interferences")
else:
    print(str(bs) + " spectra have been discarded because have atmospheric interferences")







import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path

# Print the first 5 rows of the matrix
print("Sample entries from matrix:")
for row in matrix[:5]:
    print(row)

wns = [3855, 3400, 3275, 2922, 2853, 1900, 1745, 1635, 1539, 1457, 1306, 1154, 1076, 1027, 880, 525]
print("Initial wns:", wns)
wns.sort(reverse=True)  # Ensure wavenumber values are sorted from higher to lower
print("Sorted wns:", wns)

# Now we check the lowest and highest measured wavenumber values
a = max(matrix[i][0][0] for i in range(len(matrix)) if matrix[i])
b = min(matrix[i][0][1] for i in range(len(matrix)) if matrix[i])
print("Highest measured wavenumber (a):", a)
print("Lowest measured wavenumber (b):", b)

# If only two peaks remain, einselechta prevents the algorithm from interpreting them as a range.
einselechta = len(wns) != 2

# Now we correct the wavenumber values selected that are bigger than our highest measured wavenumber
if wns[0] > a:
    if len(wns) == 2:
        wns[0] = int(a)
    else:
        while wns[0] > a:
            wns.pop(0)
        if len(wns) == 1 or wns[0] < b:
            wns.insert(0, int(a))

# And we do the same with the smaller wavenumber values:
if wns[-1] < b:
    if len(wns) == 2:
        wns[-1] = int(b)
    else:
        while wns[-1] < b:
            wns.pop()
            if len(wns) == 1:
                wns.append(int(b))
print("Adjusted wns:", wns)

simba = [m for m, n in enumerate(matata) if n == "Cat"]
sel = [["AG", "AA", "AC"], ["AG", "AA", "AC"]]
a = time.time()  # Start the timer
fida, csc, ssel = [], 0, 0
sel = [hakuna[simba[m]] if n[0].lower() == "all" else n for m, n in enumerate(sel)]
print("Selections (sel):", sel)

# Adjust wavenumber values
kk = next(i for i in range(len(matrix)) if matrix[i])
if len(wns) == 2 and not einselechta:
    resolution = 2
    wns[0] = min(wns[0], int(matrix[kk][0][0]))
    wns[-1] = max(wns[-1], int(matrix[kk][0][1]) + 1)
    wns = list(range(wns[0], wns[1], -resolution))
print("Final wns:", wns)

# Extract and filter data
for i in tqdm(matrix):
    if i:
        # Check if the spectrum matches the selection criteria
        selection_check_values = [i[2], i[3]]  # Assuming sel checks against the 3rd and 4th elements
        truth = all(any(s in sel[j] for s in selection_check_values) for j in range(len(sel)))

        # Check if the spectrum's wavenumber range matches the selected range
        if truth and i[0][0] >= wns[-1] and i[0][1] <= wns[0]:
            pos = [int(round(((i[0][2] - 1) / (i[0][1] - i[0][0])) * (j - i[0][0])) + 1) - 1 for j in wns]
            lint = [i[1][k] for k in pos]
            fuzz = [i[2 + k] for k in range(len(mbuni))]
            if len(kifaru) == 2:
                fuzz.append(str(int(i[2 + len(mbuni)])))
            fida.append(fuzz + lint)
            ssel += 1
        else:
            csc += 1

fida = sorted(fida)
fluf = [f"{matata[i]}{i + 1}" if matata[i] == "Cat" else matata[i] for i in mbuni]
if len(kifaru) == 2:
    fluf.append("StoTime")
fida.insert(0, fluf + wns)

if csc > 0:
    print(f"{csc} spectrum{' has' if csc == 1 else 's have'} been discarded because {'was' if csc == 1 else 'were'} shorter than the selected wavenumber values")

b = time.time()
print(f"This last process has lasted {round(b - a, 3)} s. The new matrix contains {ssel} spectra.")

# Convert the data to a pandas DataFrame
df = pd.DataFrame(fida[1:], columns=fida[0])

output_file_path = Path('/Users/creer/PycharmProjects/CV_Projects/.venv/Mosquito_analysis/Data/matrix.csv')

# Save the DataFrame to a CSV file
df.to_csv(output_file_path, index=False)

print(f"Matrix has been saved to {output_file_path}.")

