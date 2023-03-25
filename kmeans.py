# 时间: 2023-03-26 1:48
import pandas as pd#导入pandas库
import numpy as np
Strikers=pd.read_excel(r"C:\Users\jim\Desktop\Players Statistics.xlsx",sheet_name="Strikers")
Strikers.head()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
cols = Strikers.iloc[:, 3:9]
scaler.fit(cols)
X=pd.DataFrame(scaler.transform(cols))
X.columns=["XGoals","Goals","possession_loss","Assists","duels_won","(Goals-XGoals)/Xgoals"]
X.index=Strikers.Name
print(X)
%matplotlib notebook
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
fa=FactorAnalyzer(method="ml",rotation='varimax')
fa.fit(X)
ev,v=fa.get_eigenvalues()
plt.scatter(range(1,X.shape[1]+1),ev)
plt.plot(range(1,X.shape[1]+1),ev)
plt.title('Optimal number of clusters')
plt.xlabel('Number of clusters k')
plt.ylabel('Total Within Sum of Square')
plt.grid()
plt.show()
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
np.random.seed(1000)
km=KMeans(n_clusters=3).fit(X)
Xtype=pd.DataFrame(km.labels_)
Xtype.index=X.index
Xtype.columns=['Type']
Xnew=pd.merge(X,Xtype,left_index=True,right_index=True)
Xnew.sort_values('Type')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib notebook
fig=plt.figure()
ax1=Axes3D(fig)
x=np.array(Xnew.Goals)
y=np.array(Xnew.Assists)
z=np.array(Xnew.duels_won)
t=np.array(Xnew.Type)
ax1.scatter3D(x,y,z,c=t)
ax1.set_xlabel("Goals")
ax1.set_ylabel("Assists")
ax1.set_zlabel("duels_won")
goal=X['Goals']
assist=X['Assists']
duels_won=X['duels_won']
name=['L.M','C.R','K.M','Ney','T.M','S.M','S.G','M.R','V.O','K.K','L.M','R.L','R.L','P.D','D.V','F.C','K.H','E.H','M.R','B.S','H.K','H.M.S','M.S','D.N','R.L','O.D','K.B','Vin','Rod','A.G']
for i in range(len(goal)):
    ax1.text(goal[i],assist[i],duels_won[i],name[i])
plt.show()
np.round(Xnew.groupby(['Type'])['Goals','Assists','duels_won'].mean(),2)


import pandas as pd#导入pandas库
import numpy as np
Midfielders=pd.read_excel(r"C:\Users\jim\Desktop\Players Statistics.xlsx",sheet_name="Midfielders")
Midfielders.head()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
cols = Midfielders.iloc[:, 3:6]
scaler.fit(cols)
X=pd.DataFrame(scaler.transform(cols))
X.columns=["Accurate_balls","Key_passes","Successful_dribbles"]
X.index=Midfielders.Name
print(X)
%matplotlib notebook
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
fa=FactorAnalyzer(method="ml",rotation='varimax')
fa.fit(X)
ev,v=fa.get_eigenvalues()
plt.scatter(range(1,X.shape[1]+1),ev)
plt.plot(range(1,X.shape[1]+1),ev)
plt.title('Optimal number of clusters')
plt.xlabel('Number of clusters k')
plt.ylabel('Total Within Sum of Square')
plt.grid()
plt.show()
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
np.random.seed(1000)
km=KMeans(n_clusters=2).fit(X)
Xtype=pd.DataFrame(km.labels_)
Xtype.index=X.index
Xtype.columns=['Type']
Xnew=pd.merge(X,Xtype,left_index=True,right_index=True)
Xnew.sort_values('Type')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib notebook
fig=plt.figure()
ax1=Axes3D(fig)
x=np.array(Xnew.Accurate_balls)
y=np.array(Xnew.Key_passes)
z=np.array(Xnew.Successful_dribbles)
t=np.array(Xnew.Type)
ax1.scatter3D(x,y,z,c=t)
ax1.set_xlabel("Accurate_balls")
ax1.set_ylabel("Key_passes")
ax1.set_zlabel("Successful_dribbles")
accurate_balls=X['Accurate_balls']
key_passes=X['Key_passes']
successful_dribbles=X['Successful_dribbles']
name=['J.K','J.M','J.B','M.V','B.S','P.F','Rod','K.D.B','N.G.K','E.F','M.Ø','T.P','Jor','B.F','Cas','Fab','P.E.H','B.G','P.Z','A.F.Z.A','S.M.S','N.B','S.T','A.R','Ped','Gav','F.d.J','F.V','T.K','L.M']
for i in range(len(accurate_balls)):
    ax1.text(accurate_balls[i],key_passes[i],successful_dribbles[i],name[i])
plt.show()
np.round(Xnew.groupby(['Type'])['Accurate_balls','Key_passes','Successful_dribbles'].mean(),2)

import pandas as pd#导入pandas库
import numpy as np
Defenders=pd.read_excel(r"C:\Users\jim\Desktop\Players Statistics.xlsx",sheet_name="Defenders")
Defenders.head()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
cols = Defenders.iloc[:, 3:7]
scaler.fit(cols)
X=pd.DataFrame(scaler.transform(cols))
X.columns=["Successful_tackles","Interceptions","Clearances","Blocks"]
X.index=Defenders.Name
print(X)
%matplotlib notebook
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
fa=FactorAnalyzer(method="ml",rotation='varimax')
fa.fit(X)
ev,v=fa.get_eigenvalues()
plt.scatter(range(1,X.shape[1]+1),ev)
plt.plot(range(1,X.shape[1]+1),ev)
plt.title('Optimal number of clusters')
plt.xlabel('Number of clusters k')
plt.ylabel('Total Within Sum of Square')
plt.grid()
plt.show()
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
np.random.seed(1000)
km=KMeans(n_clusters=3).fit(X)
Xtype=pd.DataFrame(km.labels_)
Xtype.index=X.index
Xtype.columns=['Type']
Xnew=pd.merge(X,Xtype,left_index=True,right_index=True)
Xnew.sort_values('Type')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib notebook
fig=plt.figure()
ax1=Axes3D(fig)
x=np.array(Xnew.Successful_tackles)
y=np.array(Xnew.Clearances)
z=np.array(Xnew.Blocks)
t=np.array(Xnew.Type)
ax1.scatter3D(x,y,z,c=t)
ax1.set_xlabel("Successful_tackles")
ax1.set_ylabel("Clearances")
ax1.set_zlabel("Blocks")
successful_tackles=X['Successful_tackles']
clearances=X['Clearances']
blocks=X['Blocks']
name=['M.d.L','J.C','D.U','A.D','N.S','J.G','Mar','A.H','R.A','J.K','A.R','E.M','D.A','J.G','Gab','W.S','N.A','R.D','V.v.D','T.A.A','R.J','T.S','W.F','L.M','R.V','C.R','M.J.K','M.S','A.B','T.H']
for i in range(len(successful_tackles)):
    ax1.text(successful_tackles[i],clearances[i],blocks[i],name[i])
plt.show()
np.round(Xnew.groupby(['Type'])['Successful_tackles','Clearances','Blocks'].mean(),2)

import pandas as pd#导入pandas库
import numpy as np
Goalkeepers=pd.read_excel(r"C:\Users\jim\Desktop\Players Statistics.xlsx",sheet_name="Goalkeepers")
Goalkeepers.head()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
cols = Goalkeepers.iloc[:, 3:8]
scaler.fit(cols)
X=pd.DataFrame(scaler.transform(cols))
X.columns=["xG_On_Target_Conceded","Goals_conceded","Saves_percentage","Clean_sheets","chen_quation"]
#chen_quation=[(xG_On_Target_Conceded)-(Goals_conceded)]/xG_On_Target_Conceded
X.index=Goalkeepers.Name
print(X)
%matplotlib notebook
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
fa=FactorAnalyzer(method="ml",rotation='varimax')
fa.fit(X)
ev,v=fa.get_eigenvalues()
plt.scatter(range(1,X.shape[1]+1),ev)
plt.plot(range(1,X.shape[1]+1),ev)
plt.title('Optimal number of clusters')
plt.xlabel('Number of clusters k')
plt.ylabel('Total Within Sum of Square')
plt.grid()
plt.show()
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
np.random.seed(1000)
km=KMeans(n_clusters=2).fit(X)
Xtype=pd.DataFrame(km.labels_)
Xtype.index=X.index
Xtype.columns=['Type']
Xnew=pd.merge(X,Xtype,left_index=True,right_index=True)
Xnew.sort_values('Type')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib notebook
fig=plt.figure()
ax1=Axes3D(fig)
x=np.array(Xnew.Saves_percentage)
y=np.array(Xnew.Clean_sheets)
z=np.array(Xnew.chen_quation)
t=np.array(Xnew.Type)
ax1.scatter3D(x,y,z,c=t)
ax1.set_xlabel("Saves_percentage")
ax1.set_ylabel("Clean_sheets")
ax1.set_zlabel("chen_quation")
saves_percentage=X['Saves_percentage']
clean_sheets=X['Clean_sheets']
ratio=X['chen_quation']
#ratio->减少进球的程度
name=['M.N','Y.S','G.K','J.B','K.T','G.D','A.N','K.S','A.M','I.P','A.O','C.T','R.P','W.S','A.R','E.M','A.B','D.D.G','H.L','N.P','K.A','E,M','D.R','D.H','t.S','T.C','J.O','Y.B','O.V','D.C']
for i in range(len(saves_percentage)):
    ax1.text(saves_percentage[i],clean_sheets[i],ratio[i],name[i])
plt.show()
np.round(Xnew.groupby(['Type'])['Saves_percentage','Clean_sheets','chen_quation'].mean(),2)
