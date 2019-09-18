import numpy as np
import lmfit
from lmfit import Minimizer, Parameters, report_fit
import matplotlib
import matplotlib.pyplot as plt

sets = np.arange(1,14)
Es = np.array([.7,.8,1.2,2.7,5.4,9.5,14.7,19.9,24.0,28.4,31.1,33.4,34.2])

def fcn2min(params,x,data,plot_fit=False):
	a = params['a']
	b = params['b']
	c = params['c']
	d = params['d']
	if plot_fit == False:
		###good
		model = a/(1+np.exp(-c*(x-d)))+b
		###bad
		#model = a*np.tanh(c*(x-d))+b

		#model = a*np.atan((c*(x-d)))+b

		return model - data
	else:
		x_plot = np.linspace(np.min(x), np.max(x), 200)
		model = a/(1+np.exp(-c*(x_plot-d)))+b
		#model = a*np.tanh(c*(x_plot-d))+b
		return (x_plot,model)

def fit_yag(x,y):
	params = Parameters()
	params.add('a',value=15,min=0,max=50,vary=True)
	params.add('b',value=1,min=0,max=13,vary=True)
	params.add('c',value=.01,min=0,max=10,vary=True)
	params.add('d',value=7,min=0,max=13,vary=True)

	minner = Minimizer(fcn2min,params,fcn_args=(x,y))
	result = minner.minimize()
	con_report = lmfit.fit_report(result.params)
	(x_plot, model) = fcn2min(result.params, x, y, plot_fit = True)
	return (x_plot,model,result)

xfit = 0
yfit = 0
(xfit,yfit,result)=fit_yag(sets,Es)


#print(yfit)
print('a: ',result.params['a'].value)
print('b: ',result.params['b'].value)
print('c: ',result.params['c'].value)
print('d: ',result.params['d'].value)

plt.figure()
plt.scatter(sets,Es)
plt.plot(xfit,yfit)
plt.xlabel('Tick Setting')
plt.ylabel('Energy (mJ)')
plt.show()