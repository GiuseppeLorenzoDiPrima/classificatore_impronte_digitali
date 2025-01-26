:: Upgrade pip if available
python.exe -m pip install --upgrade pip

:: Install the requirements
pip install -r requirements.txt

:: Copy the dataset from a github repository
git clone https://github.com/GiuseppeLorenzoDiPrima/dataset.git