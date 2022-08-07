![Screenshot 2022-06-23 at 17 46 54](https://user-images.githubusercontent.com/24256554/175342928-f1d2af23-a5e5-436c-a8ad-a56835e99091.png)

This is the code repository for **Pricing and Trading Interest Rate Derivatives (3rd Edition)**. This repository 
is minimalist for pedagogical purposes. It does not contain documentation or code comments since all explanation 
and every step of the construction process is discussed in relevant sections of the text material. This code 
is released under a GNU Public License v3. Please review this with particular regard to sections 15 and 16 on 
warranty and liability, of which the author offers none and assumes none respectively.

### Structure of this repository

1) **modules**: this folder contains the code files that are discussed and created within the text.
2) **notebooks**: this folder contains Jupyter Notebooks that can be executed to recreate many of the examples documented in the text, often with additional aids.
3) **files**: this folder contains files that are cited in the bibliography of the text.
4) **previous_edition_material**: this folder contains legacy files from previous editions of the book, mostly Excel workbooks with former examples.
5) **tests**: this folder contains test scripts to ensure the functions continue to operate correctly as development occurred. Tests should not be considered exhaustive.
6) **requirement.txt**: this file contains the Python packages and their versions that were used in the creation of this repository.

### How to use this repository

There are three ways I envisage any user will want to use this repository:

1) As a **Casual Reader** who is interested in using the book's material and examples but not necessarily in the codebase itself. 
2) As a **New Learner** following along with the code creation and examples in the text.
3) As a **Developer** taking the codebase in the repository and repurposing it entirely.

### The Casual Reader

The casual reader who does not want to download or clone this repo can install the 
code package here from PyPI into their own Python environment using `pip`:

```commandline
~$ pip install bookirds3 
```

This is enough to execute the code examples in the book. For example the
first instance, of chapter 11 can be replicated as follows

```python
from bookirds.curves import Curve
from datetime import datetime

curve = Curve(interpolation="log_linear", nodes={
    datetime(2022, 1, 1): 1.00,
    datetime(2022, 4, 1): 0.9975,
    datetime(2022, 7, 1): 0.9945,
})
print(curve)
```

Additionally, the casual reader can download any of the Jupyter Notebooks
and the `module_loader.py` file in the same directory and execute them,
provided the installed dependencies are installed,

```commandline
(yourpythonenv) ~$ pip install pandas matplotlib jupyterlab
```



### Developers and New Learners

Developers and new learners should either **download** or **clone** this repo

```commandline
~$ git clone https://github.com/attack68/book_irds3.git
```


It is advisable to create and activate a virtual python environment, and to install of the 
necessary (and optional) dependencies.

```commandline
book_irds3$ python3 -m venv venv
book_irds3$ . venv/bin/activate
(venv) book_irds3$ pip install -r requirements.txt
```

All of the example notebooks should then be executable without errors, having started
Jupyter Lab,

```commandline
(venv) book_irds3$ jupyter lab
```

For a developer to run the test suite simply execute

```commandline
(venv) book_irds3$ pytest tests
```