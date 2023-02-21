This is an overview of how to create an .exe file from the .py app code.
open the documentation via this link:

https://docs.google.com/document/d/1MUWn_lzMKkZLdb-U0IrUpYRCmB-wr_yfPokJv0ps4Ns/edit?usp=sharing

===================Quick copy of documentation without images (not recommended)================
Create virtual environment in cmd

You can make a virtual environment and set up dependancies like the python version and packages. Then you can share the whole thing by copying it. Before you start coding you need to activate the specific environment you want to work in.

The location of the venv is not per se linked to your .py script file. You activate your environment and then start running .py files from any location.

conda create --name venv1   (in anaconda promt)

Or

 conda create --name venv1 python=3.6  (for specifying version)

conda activate venv1

Or when finished:

conda deactivate

List all environments:

Conda env list

To view what is installed in your virtual environment:

conda list

The environment can be seen in anaconda also. When there is a play button it means there the venv is activated.

Conda install packagename==version

Conda install numpy


Interpreters such as jupyter notebook and pycharm are installed within the venv.


Running Jupyter notebook from a virtual environment (created in anaconda)

Start anaconda prompt:
Activate the environment

Conda activate namevenv

To use jupyter notebook we need to install ipykernel in the virtual environment first.

conda install -c anaconda ipykernel

Now we need to add the venv to jupyter notebook

python -m ipykernel install --user --name=venv1



Now Lauch jupyter notebook and make a script .py file at any location. At top menu: Kernel, change kernel. → choose the venv name you created earlier.

You can also start jupyternotebook by:
Jupyter.exe notebook

Download geemap (googleearth engine for python)

In anaconda prompt:
conda install -c conda-forge geemap

Download other packages

conda install geopandas

#Packages could also be installed with mamba in the conda prompt, its a bit quicker than #conda installer…
#conda install mamba -c conda-forge

#Mamba install geemap xxarray_leaflet -c conda-forge

Download a tutorial of geemap from its github documentation & run from conda prompt

https://github.com/giswqs/geemap/tree/master/examples/notebooks

Download example: 39 (scroll down, note tutorial 41 is very useful templet for app building!)
Click on ipynb file in github, click right on ‘raw’, safe link as. Ipynb file, Safe in downloads (anywhere).

Just open this notebook in Jupyter notebook

#In the conda prompt, (having activated the right environment see above),
#cd Downloads

#Look at files that are there
#dir

#Type:
#jupyter notebook
#This will open jupyter notebook at location you cd’t to.

Working directory for project
import os
os.chdir(path)

os. getcwd()


remove a conda env

conda env remove --name envnamehereac

Connect virtual environment from miniforge to pycharm community

Create virtual env just as in normal conda.
Start pycharm community. New project. Then bottom right conner right click new interpreter. Navigate to conda environment.
At conda executable enter something like

C:\Users\mathijs\AppData\Local\miniforge3\_conda.exe

Behind interpreter: browse to e.g.:
C:\Users\mathijs\AppData\Local\miniforge3\envs\py39envname\python.exe
You might need to enter AppData in the type box as this folder is hidden.


‘Location’ is location of venv  so… C:\Users\mathijs\.conda\envs\’excisting venvname here’

conda env remove --name pythonProject

View imported packages
conda list

Cmd navigation

CD\. It takes you to the top of the directory tree
Cd .. go up
DIR View contenst

Make exe file

Pip install pyinstaller in conda prompt (better to install it directly where your .py folder is as you use pyinstaller within that folder).

Find the pyinstaller exe in your conda created environment
C:\Users\mathijs\AppData\Local\miniforge3\envs\py39\Scripts   py39 is the environment folder

Copy exe to the same folder where your .py script is located.

Content folder:

Now navigate to the folder where the .py is located in cmd

Now run pyinstaller.exe with setup commands (check documentation) such as:

 C:\Users\mathijs\PycharmProjects\pythonProject>pyinstaller.exe --onefile --icon=gns.ico
GNSRechargeApp.py
If you don’t want the concol, add –noconsole
(mind spaces are important before --)
Woop woop… program will be created.

In the dist folder there is an exe file now. All other stuff can be deleted: .ida build getestprm1.spec

Note: Your icon file and everything should be in the same folder.
Designate an icon file.













