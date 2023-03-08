import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

[(f.name, f.fname) for f in fm.fontManager.ttflist if 'Nanum' in f.name]


print(mpl.rcParams['font.family']) # font
print(mpl.rcParams['font.size']) # size