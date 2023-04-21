python3 -m pip install --upgrade pip
python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv

python3 -m virtualenv ft_env
source ft_env/bin/activate

pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install mne