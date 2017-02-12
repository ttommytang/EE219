import numpy as np
import pandas as pd
import scipy as sp
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# --------------- Load the data from the csv file and do pre-processing --------------------
boston = pd.read_csv("housing_data.csv")
boston.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'BP', 'LSTAT',
                  'MEDV']
bostonDesign = boston.loc[:, 'CRIM':'LSTAT']
bostonTarget = boston.loc[:, 'MEDV']

# -------------------------------- Linear regresion ----------------------------------------
ln = linear_model.LinearRegression()
ln.fit(bostonDesign, bostonTarget)

bostonLNCoef = ln.coef_
bostonLNMSE = np.mean((ln.predict(bostonDesign) - bostonTarget) ** 2)


def calculate_rmse(predicted, actual):
    return sp.sqrt(sp.mean((predicted - actual) ** 2))


# -------------------------------- Cross Validation ---------------------------------------
def cv_analysis(model, design, target, cv):
    predicted = cross_val_predict(model, design, target, cv=cv)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    ax1.scatter(target, predicted)
    ax1.plot([target.min(), target.max()], [target.min(), target.max()], 'k--', lw=4)
    ax1.set_xlabel('Measured', fontsize=20)
    ax1.set_ylabel('Predicted', fontsize=20)
    ax1.set_title('Fitted values vs. Actual values', fontsize=20)

    ax2.scatter(predicted, predicted - target)
    ax2.set_xlabel('Predicted', fontsize=20)
    ax2.set_ylabel('Residual', fontsize=20)
    ax2.axhline(y=0, c="green", linewidth=1.5, zorder=0)
    ax2.set_title('Residuals vs. Fitted value', fontsize=20)
    plt.show()

    cv_rmse = calculate_rmse(predicted, target)
    return predicted, cv_rmse


cv_predicted, cv_rmse = cv_analysis(ln, bostonDesign, bostonTarget, 10)

# CV_RMSE = calculate_RMSE(bostonCV, bostonTarget)
print "RMSE for the 10-fold CV = " + str(cv_rmse)

# -------------------------------- Significance of different attributes  ---------------------------------------
AttrMSE = []
for attr in bostonDesign.columns:
    train = bostonDesign.copy()
    train.drop(attr, 1, inplace=True)

    ln = linear_model.LinearRegression()
    ln.fit(train, bostonTarget)

    predicted = cross_val_predict(ln, train, bostonTarget, cv=10)
    AttrMSE.append(calculate_rmse(predicted, bostonTarget))
    print 'RMSE(' + str(attr) + ' excluded) = ' + str((calculate_rmse(predicted, bostonTarget)))

# From the result above we can see that the least significant attributes: RM, AGE and CRIM
# So we consider to drop these three attributes to see if we can get an optimized model:
train = bostonDesign.copy()
train.drop('RM', 1, inplace=True)
train.drop('AGE', 1, inplace=True)
# train.drop('CRIM', 1, inplace=True)

ln = linear_model.LinearRegression()
ln.fit(train, bostonTarget)

opt_predicted, opt_rmse = cv_analysis(ln, train, bostonTarget, 10)
print 'The RMSE after optimization according to significance of attributes = ' + str(opt_rmse)

plt.clf()
ax3 = plt.subplot()
plt.scatter(range(1, len(bostonDesign.columns) + 1, 1), AttrMSE)
plt.xlabel('Attribute excluded', fontsize=20)
plt.ylabel('RMSE', fontsize=20)
plt.title('Significance of all the attributed', fontsize=20)
plt.xticks(range(1, len(bostonDesign.columns) + 1, 1), bostonDesign.columns)
plt.axhline(y=cv_rmse, c="green", linewidth=1.5, zorder=0)
plt.axhline(y=opt_rmse, c="yellow", linewidth=1.5, zorder=0)
plt.show()

# ---------------------------------------------- Polynomial regression  -----------------------------------------
polyRMSE = []

for deg in range(1, 7, 1):
    poly = PolynomialFeatures(degree=deg, interaction_only=False)
    polyDesign = poly.fit_transform(bostonDesign)

    ln = linear_model.LinearRegression()
    ln.fit(polyDesign, bostonTarget)

    predicted = cross_val_predict(ln, polyDesign, bostonTarget, cv=10)
    temp = (calculate_rmse(predicted, bostonTarget))
    polyRMSE.append(temp)
    print 'RMSE(with ' + str(deg) + ' degree polynomial penalty function) = ' + str(temp)

plt.clf()
ax4 = plt.subplot()
plt.plot(range(1, 7, 1), polyRMSE, linewidth=1.5)
plt.xlabel('Degree of polynomial function', fontsize=20)
plt.ylabel('RMSE', fontsize=20)
plt.title('RMSE as function of Degree of Polynomial Penalty Function')

plt.show()

# ---------------------------------------------- Ridge regression  -----------------------------------------
clf = Ridge()

RidgeCoef = []
RidgeMSE = []

alphas = np.logspace(-5, 5, num=11)
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(bostonDesign, bostonTarget)
    RidgeCoef.append(clf.coef_)
    RidgePredicted = clf.predict(bostonDesign)
    temp = (calculate_rmse(RidgePredicted, bostonTarget))
    RidgeMSE.append(temp)
    print 'RMSE of Ridge(tuning factor: alpha = ' + str(a) + ') = ' + str(temp)

plt.clf()
ax5 = plt.subplot()
ax5.plot(alphas, RidgeMSE)
ax5.set_xscale('log')
plt.xlabel('alpha', fontsize=20)
plt.ylabel('RMSE', fontsize=20)
plt.title('RMSE as a function of the regularization - Ridge', fontsize=20)
plt.axis('tight')
plt.show()

# ---------------------------------------------- Lasso regression  -----------------------------------------
clf = Lasso()

LassoCoef = []
LassoMSE = []

betas = np.logspace(-5, 5, num=11)
for b in betas:
    clf.set_params(alpha=b)
    clf.fit(bostonDesign, bostonTarget)
    LassoCoef.append(clf.coef_)
    LassoPredicted = clf.predict(bostonDesign)
    temp = (calculate_rmse(LassoPredicted, bostonTarget))
    LassoMSE.append(temp)
    print 'RMSE of Lasso(tuning factor: beta = ' + str(b) + ') = ' + str(temp)

plt.clf()
ax6 = plt.subplot()
ax6.plot(betas, LassoMSE)
ax6.set_xscale('log')
plt.xlabel('beta', fontsize=20)
plt.ylabel('RMSE', fontsize=20)
plt.title('RMSE as a function of the regularization - Lasso', fontsize=20)
plt.axis('tight')
plt.show()
