===================================================================================================================================
                  Rapid age-grading and species identification of natural mosquitoes for malaria surveillance
===================================================================================================================================

Doreen J. Siria, Roger Sanou, Joshua Mitton, Emmanuel P. Mwanga, Abdoulaye Niang, Issiaka Sare, Paul C.D. Johnson, Geraldine Foster,
Adrien M.G. Belem, Klaas Wynne, Roderick Murray-Smith, Heather M. Ferguson, Mario González-Jiménez*, Simon A. Babayan*, Abdoulaye
Diabaté, Fredros O. Okumu, and Francesco Baldini*

*Corresponding authors:

Mario.GonzalezJimenez(at)glasgow.ac.uk
Simon.Babayan(at)glasgow.ac.uk
Francesco.Baldini(at)glasgow.ac.uk

LINK TO ENLIGHTEN DATABASE: https://researchdata.gla.ac.uk/1235/

-----------------------------------------------------------------------------------------------------------------------------------
                                                             Notes
-----------------------------------------------------------------------------------------------------------------------------------

* The spectra are organised in folders according to the experiment to which they correspond and the country of origin.

* The infrared spectrum of each mosquito is stored in a .mzz file. We have designed this format so that the spectra of our >40,000
mosquitoes take up as little space as possible. It consists of a zip-compressed text file in which the spectral information is
summarised in a single column. As the wavenumbers in the spectrum are sequential and spaced at equal intervals, instead of including
all of them in the file we have only indicated the starting wavenumber of the spectrum (first item in the column), the final (second
item in the column) and the number of wavenumbers measured (the third item). This way the list of measured wavenumbers can
be easily created with this formula:

                                  i * (a - b)
                    wvn_i = a - ---------------   for i from 0 to c.
                                      c 

Where:
   a -> Initial wavenumber (first item .mzz file)
   b -> Final wavenumber (second item .mzz file)
   c -> Number of measured wavenumbers (third item .mzz file)

All wavenumbers were measured in units of reciprocal centimeters (cm-1). The remaining data in the column are the measured absorbances
for each wavenumber.

* The name of each spectrum contains the basic information of the measured mosquito. 


                                           SS-C-xxD-TT-EE-yymmdd-YYMMDD-33.mzz

where:
               SS: Species code:
                      AA: Anopheles arabiensis
                      AG: Anopheles gambiae
                      AC: Anopheles coluzzi
               C: Country code:
                      B: Burkina Faso
                      S: Scotland
                      T: Tanzania
               xx: Age of the mosquito in days (UNK if unknown)
	       TT: Status of the mosquito:
                      BF: Blood fed
                      SF: Sugar fed
                      UF: Unfed
                      GR: Gravid
                      WG: Gravid (wild mosquitoes)
                      NP: Nulliparous(Gonotrophic cycle 0)
                      P1: Gonotrophic cycle 1
                      P2: Gonotrophic cycle 2
                      P3: Gonotrophic cycle 3
                      P4: Gonotrophic cycle 4
                      UN: Unknown
	       EE: Experiment
                      TL: Laboratory variation
                      TF: Genetic variation
                      VF: Environmental variation
                      WV: Wild mosquitoes
               yymmdd: Date the mosquito was collected
               YYMMDD: Date the mosquito was measured
               33: Mosquito number identifier








