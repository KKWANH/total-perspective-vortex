#updateing & install virtual env
python3	-m pip install --upgrade pip
python3	-m pip install --user --upgrade pip
python3	-m pip install --user virtualenv

#setting virtual env
python3	-m virtualenv ft_env
source	ft_env/bin/activate

#updating pip version
pip install --upgrade pip

#installing python libraries
pip3 install pandas
pip3 install numpy
pip3 install matplotlib
pip3 install scikit-learn
pip3 install mne