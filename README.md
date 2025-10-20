## Summary

Prediksjon av norsk stortingsval basert på pollofpolls-data. 


## Setup on a Windows system:

For Windows system users, the easiest way is probably to install a WinPython distribution that already includes Python 3.11 and Jupyter Notebooks.

This version is tested to work with the demo and exercise: https://github.com/winpython/winpython/releases/tag/13.1.202502222final

Download WinPython and extract the files to a folder of your choice.

Open the extracted folder and run WinPython Command Prompt.exe.

Navigate to your GitHub folder with the exercise by typing cd C:\[....]\GitHub\OneSixtyNine.

## Setup on Linux/WSL and macOS:

Python comes preinstalled on most Linux distributions, and is available as a package on all others.

However, should you need to install it check out this link: https://diveintopython.org/learn/install/linux

To install python on macOS follow this link: https://diveintopython.org/learn/install/mac

With python installed, run these commands from the root of the project:
```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Shared steps:

Type `jupyter notebook` to start Jupyter Notebooks. 

Select the  notebook onesixtyninevBouvet1-13032025 in the browser window that opens.

Run the notebook. (Select Run - Run all cells).  It should run without problems and produce graphs in the bouvet one presentation. 

Note: If you prefer Jupyter Lab over Jupyter Notebook, you can install it by typing pip install jupyterlab in your command terminal and then following similar steps to launch it as done with Jupyter Notebook.


****


Pitch til Bouvet One:
***
Kan vi forutsjå kven som vinn stortingsvalet i 2025 – med data science?

I 2008 blei Nate Silver verdskjend då hans valprognosemodell korrekt føresåg utfallet i 49 av 50 statar i det amerikanske presidentvalet. Inspirert av hans metode vil vi utforske om vi kan gjere det same for Stortingsvalet 2025.

Kan vi følgje i hans fotspor og predikere utfallet av stortingsvalget i Noreg i 2025? 

Vi vil arrangere ein kodekveld senere i vår der vi vil prøve å lage ein slik modell. Dette foredraget vil legge grunnlaget og utforske hvilke data som finst offentleg tilgjengeleg, som t.d. poll-of-polls, politiske skandaler, regjeringsskifte, rentenivå og arbeidsledigheitstal - og har modifiserte Sainte-Laguës metode, utjamningsmandat og sperregrense å seie for utfallet?

Kodekveld 15.05.2025: (https://event.bouvet.no/event/e6bcce8f-d697-40c3-a8f8-136ce03cdf84)

****
Det finst eit git-repo med python-kode som berekner mandat utifrå metoden som brukast i Noreg
 
https://github.com/martinlackner/apportionment
