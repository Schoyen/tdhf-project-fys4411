Search.setIndex({"docnames": ["intro", "task-1-single-particle-basis", "task-2-rhf-solver", "task-3-expectation-values", "task-4-tdrhf-solver", "task-5-td-expectation-values"], "filenames": ["intro.md", "task-1-single-particle-basis.ipynb", "task-2-rhf-solver.ipynb", "task-3-expectation-values.ipynb", "task-4-tdrhf-solver.ipynb", "task-5-td-expectation-values.ipynb"], "titles": ["Time-dependent Hartree-Fock theory", "Task 1: Single-particle basis", "Task 2: Restricted Hartree-Fock ground state solver", "Task 3: Expectation values", "Task 4: Time-dependent restricted Hartree-Fock solver", "Task 5: Time-dependent expectation values"], "terms": {"thi": [0, 1, 2, 3, 4, 5], "method": [0, 2, 3, 4], "appli": [0, 5], "system": [0, 4], "two": [0, 1, 2, 3, 5], "one": [0, 2, 3, 4, 5], "dimension": [0, 4], "quantum": 0, "dot": [0, 1, 2, 3, 4], "cours": 0, "fys4411": 0, "2023": 0, "univers": 0, "oslo": 0, "we": [0, 1, 2, 3, 4, 5], "base": 0, "around": [0, 2], "paper": 0, "zkbs04": [0, 1, 4], "below": [0, 1], "follow": 0, "list": [0, 1], "need": [0, 1, 2, 4], "complet": 0, "set": [0, 1, 2, 4], "up": [0, 1, 2], "singl": [0, 2, 3, 5], "particl": [0, 2, 5], "basi": [0, 2, 3, 4, 5], "creat": 0, "restrict": [0, 1], "rhf": [0, 2], "ground": [0, 3, 4], "state": [0, 1, 3, 4, 5], "solver": 0, "comput": [0, 1, 3, 5], "energi": [0, 1, 2], "densiti": [0, 2, 5], "dipol": [0, 1, 4], "moment": [0, 4], "tdrhf": 0, "add": [0, 2, 4], "monochromat": [0, 1, 4], "laser": [0, 1, 4], "field": [0, 1, 2, 4], "interact": [0, 1, 2, 4, 5], "term": [0, 1, 2, 4], "hamiltonian": [0, 2, 4], "overlap": 0, "plot": [0, 1], "result": 0, "option": 0, "run": 0, "simul": [0, 1, 2], "puls": 0, "fourier": 0, "spectra": 0, "after": 0, "turn": [0, 2], "off": 0, "cm92": [0, 1], "daniel": 0, "t": [0, 1, 2, 4, 5], "colbert": [0, 1], "william": 0, "h": [0, 1, 2, 3, 4], "miller": [0, 1], "A": [0, 2, 5], "novel": 0, "discret": [0, 1], "variabl": [0, 1], "represent": [0, 1, 2], "mechan": 0, "reactiv": 0, "scatter": [0, 1], "via": 0, "s": [0, 2], "matrix": [0, 2, 3, 4, 5], "kohn": 0, "The": [0, 1, 4], "journal": 0, "chemic": 0, "physic": 0, "96": 0, "3": [0, 1, 2], "1982": 0, "1991": 0, "1992": 0, "doi": 0, "10": [0, 1, 2], "1063": 0, "1": [0, 2, 3, 4, 5], "462100": 0, "gri10": [0, 1], "david": 0, "j": [0, 1, 2, 3], "griffith": 0, "introduct": 0, "prentic": 0, "hall": 0, "second": 0, "edit": 0, "2010": 0, "gc93": [0, 1], "gerrit": 0, "c": [0, 2, 3, 4, 5], "groenenboom": [0, 1], "combin": 0, "99": 0, "12": 0, "9681": 0, "1993": 0, "465450": 0, "jkp09": [0, 4], "joachain": 0, "n": [0, 1, 2, 3, 5], "kylstra": 0, "r": [0, 1, 2], "m": 0, "potvlieg": 0, "atom": [0, 2, 3, 4, 5], "intens": 0, "cambridg": 0, "press": 0, "2009": 0, "1017": 0, "cbo9780511993459": 0, "zanghellini": [0, 1], "kitzler": 0, "brabec": 0, "scrinzi": 0, "test": 0, "multi": 0, "configur": 0, "b": [0, 2], "molecular": [0, 2, 3, 4, 5], "optic": 0, "37": 0, "763": 0, "2004": 0, "1088": 0, "0953": 0, "4075": 0, "4": [0, 1], "004": 0, "here": [1, 2, 4], "discuss": [1, 4], "how": [1, 2], "us": [1, 2, 4], "et": 1, "al": 1, "our": [1, 2, 4], "specif": 1, "import": 1, "numpi": 1, "np": 1, "matplotlib": 1, "pyplot": 1, "plt": 1, "hat": [1, 2, 3, 4, 5], "x_1": [1, 2], "x_n": [1, 2], "sum_": [1, 2, 3, 4, 5], "i": [1, 2, 3, 4, 5], "left": [1, 2, 4], "frac": [1, 2, 3, 4], "2": [1, 3, 4, 5], "mathrm": 1, "d": [1, 2, 4], "x_i": 1, "omega": [1, 4], "x": [1, 3, 4, 5], "_i": [1, 4], "mathcal": [1, 4], "e": [1, 2, 3, 4], "_0": [1, 4], "sin": [1, 4], "right": [1, 2], "sqrt": [1, 2], "_j": 1, "where": [1, 2, 3, 4, 5], "denot": [1, 2, 5], "time": [1, 2], "independ": [1, 2, 4, 5], "bodi": [1, 2, 3, 4], "equiv": [1, 2, 4], "v": 1, "kinet": 1, "oper": [1, 2, 4, 5], "harmon": [1, 2, 4], "oscil": [1, 2, 4], "potenti": 1, "soft": 1, "coulomb": [1, 5], "u": [1, 2, 3, 5], "x_j": 1, "semi": [1, 4], "classic": [1, 4], "describ": [1, 4], "depend": [1, 2], "begin": [1, 2, 3, 4, 5], "align": [1, 2, 3, 4, 5], "end": [1, 2, 3, 4, 5], "As": [1, 2, 4], "done": [1, 3], "ourselv": 1, "trap": 1, "frequenc": [1, 4], "0": [1, 2, 5], "25": 1, "shield": 1, "paramet": 1, "For": [1, 2, 4, 5], "strength": 1, "8": [1, 2], "full": [1, 2, 3], "so": 1, "accord": 1, "consist": [1, 2], "four": 1, "compon": 1, "thei": [1, 3], "ar": [1, 2, 3], "an": [1, 2, 4, 5], "orthonorm": [1, 2, 4, 5], "chi_": 1, "alpha": 1, "mid": [1, 2], "k": [1, 2, 5], "real": [1, 2], "point": 1, "quadratur": 1, "weight": 1, "q": [1, 2, 4, 5], "x_": 1, "w_": 1, "properti": 1, "beta": 1, "delta_": [1, 2], "rule": 1, "all": [1, 2], "multipl": 1, "within": 1, "approxim": [1, 2, 4], "dub": 1, "function": [1, 2, 3, 4, 5], "note": [1, 2, 4], "even": [1, 2], "though": 1, "when": [1, 2, 3], "necessarili": [1, 2, 4], "zero": [1, 5], "valu": 1, "see": 1, "exampl": [1, 5], "That": [1, 2, 4], "doe": [1, 2], "have": [1, 2, 3, 4, 5], "everywher": 1, "except": 1, "howev": [1, 2], "orthogon": 1, "abov": 1, "given": [1, 2, 3, 5], "mathemat": 1, "langl": [1, 2, 3, 4, 5], "rangl": [1, 2, 3, 4, 5], "approx": [1, 2], "gamma": 1, "onli": [1, 2], "evalu": [1, 3], "diagon": 1, "defin": [1, 2], "delta": 1, "case": [1, 2, 4], "pi": 1, "neq": 1, "quad": 1, "x_0": 1, "uniform": 1, "grid": [1, 3], "equal": [1, 2], "arbitrari": [1, 2], "start": [1, 2], "same": [1, 4, 5], "class": 1, "sincdvr": 1, "def": 1, "__init__": 1, "self": [1, 2], "x_a": 1, "dx": 1, "__call__": 1, "return": 1, "space": 1, "x_min": 1, "5": [1, 2], "most": [1, 2], "side": 1, "box": 1, "x_max": 1, "num_dvr": 1, "int": 1, "number": [1, 2], "linspac": 1, "callabl": 1, "demonstr": 1, "what": 1, "look": [1, 2], "like": 1, "both": [1, 2, 5], "kroneck": 1, "more": [1, 2, 4], "refin": 1, "color": 1, "label": 1, "chi_3": 1, "legend": 1, "show": [1, 2], "x_3": 1, "other": 1, "between": [1, 2, 5], "exactli": 1, "from": [1, 2, 3, 4, 5], "shown": 1, "equat": [1, 4], "can": [1, 2, 3, 5], "found": [1, 3, 5], "a7": 1, "ref": 1, "t_": 1, "6": [1, 2], "remain": 1, "part": [1, 2], "integr": 1, "which": [1, 2, 4], "solv": [1, 2, 4], "posit": [1, 5], "bra": 1, "ket": 1, "pairwis": 1, "x_2": 1, "y": 1, "twice": 1, "get": [1, 2, 4], "express": [1, 2], "onc": 1, "you": 1, "check": [1, 4], "setup": 1, "ha": [1, 2], "been": 1, "correctli": 1, "eigenvalu": [1, 2], "should": 1, "correspond": 1, "chapter": [1, 4], "differ": [1, 2], "much": 1, "might": 1, "increas": 1, "lowest": [1, 2], "correct": 1, "schr\u00f6dinger": 2, "psi": [2, 3, 4], "mani": [2, 3, 4], "wave": [2, 4], "know": 2, "non": 2, "exact": 2, "slater": [2, 5], "determin": [2, 5], "phi": [2, 3, 4, 5], "eigenst": 2, "motiv": 2, "ansatz": [2, 3, 4], "phi_1": [2, 3, 4], "phi_2": [2, 4], "phi_n": [2, 3, 4], "mo": 2, "also": [2, 3], "known": [2, 3, 4], "phi_p": [2, 4], "p": [2, 4, 5], "l": [2, 3, 4, 5], "leq": 2, "primari": 2, "unknown": 2, "otim": 2, "cdot": 2, "product": 2, "first": 2, "anti": 2, "symmetr": 2, "make": 2, "convent": 2, "split": 2, "occupi": [2, 5], "virtual": 2, "_": [2, 3, 4, 5], "phi_i": [2, 3], "cup": 2, "phi_a": 2, "let": [2, 4], "indic": 2, "contain": 2, "refer": 2, "find": [2, 4], "minim": 2, "closest": 2, "true": 2, "e_": 2, "gs": 2, "variat": 2, "principl": 2, "subject": 2, "constraint": 2, "phi_j": [2, 3], "ij": 2, "impli": [2, 4], "normal": 2, "To": 2, "includ": 2, "requir": 2, "lagrang": 2, "undetermin": 2, "multipli": 2, "lambda": 2, "lambda_": 2, "ji": 2, "each": 2, "lagrangian": 2, "gener": 2, "hermitian": 2, "well": [2, 4], "notat": 2, "befor": 2, "proceed": 2, "decid": 2, "henc": 2, "intend": 2, "In": [2, 4], "particular": 2, "wish": [2, 3, 5], "treat": 2, "spin": 2, "electron": [2, 3, 4], "singlet": 2, "work": 2, "guarante": 2, "solut": 2, "z": 2, "project": [2, 3, 4], "m_": 2, "pm": 2, "breviti": 2, "sigma": 2, "down": 2, "abl": 2, "control": 2, "symmetri": 2, "typic": 2, "three": 2, "kind": 2, "form": 2, "a_": 2, "varphi_": 2, "p0": 2, "p1": 2, "spatial": 2, "direct": 2, "coeffici": [2, 3, 4, 5], "alpha_": 2, "ensur": 2, "overal": 2, "unrestrict": 2, "lfloor": 2, "rfloor": 2, "definit": 2, "allow": 2, "integ": 2, "divis": 2, "doubli": 2, "name": 2, "bit": 2, "mislead": 2, "relat": 2, "common": 2, "ghf": 2, "care": 2, "about": 2, "mean": [2, 4], "_z": 2, "nor": 2, "uhf": 2, "give": [2, 5], "final": 2, "opt": 2, "assum": [2, 4], "varphi_i": 2, "varphi_j": 2, "insert": [2, 3, 4], "7": 2, "coordin": 2, "write": 2, "vmatrix": 2, "vdot": 2, "ddot": 2, "deriv": [2, 4], "canon": 2, "yield": 2, "f": [2, 4], "varepsilon_p": 2, "limit": 2, "element": [2, 3, 4, 5], "xi": 2, "rangle_": [2, 3], "AS": [2, 3], "wai": [2, 5], "expand": [2, 4], "psi_": [2, 3, 4, 5], "mu": [2, 3, 4, 5], "take": 2, "doubl": 2, "onto": 2, "consequ": 2, "truncat": 2, "computation": 2, "complex": 2, "post": 2, "g": 2, "coupl": 2, "cluster": 2, "lower": 2, "attent": 2, "transform": [2, 5], "ao": 2, "c_": [2, 3, 4, 5], "now": [2, 4], "becom": 2, "gather": [2, 4], "varepsilon_": 2, "nu": [2, 3, 4, 5], "f_": [2, 4], "mathbf": 2, "boldsymbol": [2, 5], "varepsilon": 2, "varepsilon_1": 2, "varepsilon_k": 2, "vector": 2, "eigenenergi": 2, "confus": 2, "last": 2, "construct": [2, 3, 4], "replac": 2, "sum": 2, "over": 2, "kappa": 2, "h_": 2, "d_": [2, 3, 5], "complic": 2, "factor": 2, "techniqu": 2, "call": 2, "iter": 2, "There": 3, "main": 3, "quantiti": 3, "rho": [3, 5], "orbit": [3, 4, 5], "optim": 3, "expans": [3, 4], "matric": 3, "alreadi": [3, 5], "problem": 4, "out": 4, "shine": 4, "append": 4, "electr": 4, "length": 4, "gaug": 4, "thorough": 4, "charg": 4, "alwai": 4, "activ": 4, "kept": 4, "evolut": 4, "text": 4, "hbar": 4, "chosen": 4, "eigenfunct": 4, "choos": 4, "occur": 4, "t_0": 5, "bigl": 5, "bigr": 5, "det": 5, "dagger": 5, "_o": 5, "c_o": 5, "evolv": 5, "o": 5, "column": 5, "act": 5, "wherea": 5, "static": 5, "hartre": 5, "fock": 5, "altern": 5, "psi_p": 5, "phi_": 5, "phi_q": 5, "rho_": 5, "pmatrix": 5, "ident": 5, "block": 5, "elsehwer": 5}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"time": [0, 4, 5], "depend": [0, 4, 5], "hartre": [0, 2, 3, 4], "fock": [0, 2, 3, 4], "theori": 0, "project": 0, "task": [0, 1, 2, 3, 4, 5], "1": 1, "singl": 1, "particl": [1, 3], "basi": 1, "hamiltonian": 1, "definit": 1, "one": 1, "dimension": 1, "dvr": 1, "sinc": 1, "matrix": 1, "element": 1, "problem": [1, 2], "b": 1, "tip": 1, "2": 2, "restrict": [2, 3, 4], "ground": 2, "state": 2, "solver": [2, 4], "type": 2, "orbit": 2, "system": 2, "The": [2, 3], "roothan": 2, "hall": 2, "equat": 2, "3": 3, "expect": [3, 5], "valu": [3, 5], "energi": 3, "densiti": 3, "4": 4, "5": 5, "overlap": 5, "One": 5, "bodi": 5}, "envversion": {"sphinx.domains.c": 2, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 6, "sphinx.domains.index": 1, "sphinx.domains.javascript": 2, "sphinx.domains.math": 2, "sphinx.domains.python": 3, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx.ext.intersphinx": 1, "sphinxcontrib.bibtex": 9, "sphinx": 56}})