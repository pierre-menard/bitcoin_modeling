#https://towardsdatascience.com/efficient-time-series-using-pythons-pmdarima-library-f6825407b7f0

import pandas as pd
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time



df=pd.read_csv('paris/testLong.csv')

print(df.head(3))


from pmdarima.model_selection import train_test_split

train_len = int(df.shape[0] * 0.8)
train_data, test_data = train_test_split(df, train_size=train_len)

y_train = train_data['Hash'].values
y_test = test_data['Hash'].values

print(f"{train_len} train samples")
print(f"{df.shape[0] - train_len} test samples")

from pandas.plotting import lag_plot


'''
fig, axes = plt.subplots(3, 2, figsize=(8, 12))
plt.title('Autocorrelation plot')

# The axis coordinates for the plots
ax_idcs = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (2, 0),
    (2, 1)
]

for lag, ax_coords in enumerate(ax_idcs, 1):
    ax_row, ax_col = ax_coords
    axis = axes[ax_row][ax_col]
    lag_plot(df['Hash'], lag=lag, ax=axis)
    axis.set_title(f"Lag={lag}")

#plt.tight_layout()
'''

from pmdarima.arima import ndiffs

#1007 6288 1257 3775


kpss_diffs = ndiffs(y_train, alpha=0.05, test='kpss', max_d=6)
adf_diffs = ndiffs(y_train, alpha=0.05, test='adf', max_d=6)
n_diffs = max(adf_diffs, kpss_diffs)

print(f"Estimated differencing term: {n_diffs}")


auto = pm.auto_arima(y_train, d=n_diffs, seasonal=False, stepwise=True,
                     suppress_warnings=True, max_p=6, trace=2)

print(auto.order)

from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape

model = auto  # seeded from the model we've already fit

def forecast_one_step():
    fc, conf_int = model.predict(n_periods=1, return_conf_int=True)
    return (
        fc.tolist()[0],
        np.asarray(conf_int).tolist()[0])

forecasts = []
confidence_intervals = []

for new_ob in y_test:
    fc, conf = forecast_one_step()
    forecasts.append(fc)
    confidence_intervals.append(conf)

    # Updates the existing model with a small number of MLE steps
    model.update(new_ob)

print(f"Mean squared error: {mean_squared_error(y_test, forecasts)}")
print(f"SMAPE: {smape(y_test, forecasts)}")


fig, axes = plt.subplots(2, 1, figsize=(12, 12))

# --------------------- Actual vs. Predicted --------------------------
axes[0].plot(y_train, color='blue', label='Training Data')
axes[0].plot(test_data.index, forecasts, color='green', marker='o',
             label='Predicted Price')

axes[0].plot(test_data.index, y_test, color='red', label='Actual Price')
axes[0].set_title('Prediction')
axes[0].set_xlabel('Dates')
axes[0].set_ylabel('Prices')

#axes[0].set_xticks(np.arange(0, 7982, 1300).tolist(), df['Month'][0:7982:1300].tolist())
axes[0].legend()


# ------------------ Predicted with confidence intervals ----------------
axes[1].plot(y_train, color='blue', label='Training Data')
axes[1].plot(test_data.index, forecasts, color='green',
             label='Predicted Hash')

axes[1].set_title('Prices Predictions & Confidence Intervals')
axes[1].set_xlabel('Dates')
axes[1].set_ylabel('Prices')

conf_int = np.asarray(confidence_intervals)
axes[1].fill_between(test_data.index,
                     conf_int[:, 0], conf_int[:, 1],
                     alpha=0.9, color='orange',
                     label="Confidence Intervals")

#axes[1].set_xticks(np.arange(0, 7982, 1300).tolist(), df['Date'][0:7982:1300].tolist())
axes[1].legend()

plt.show()


print ("********")

time.sleep (10)











DATA_PATH = "paris/testLong.csv"

df=pd.read_csv('paris/testLong.csv')
df=df.rename(columns={'Hash':'hash','Month':'date'})
df['date'] = pd.to_datetime(df['date'])
df.set_index(df['date'], inplace=True)
df=df.drop(columns=['date'])
df = df.drop(columns=['Addresses'])
df = df.drop(columns=['Difficulty'])
df.head()

df = df[3000:]


adf_test=pm.arima.ADFTest(alpha=0.05)
adf_test.should_diff(df)# Output
(0.01, False)

train=df[:1600]
test=df[-400:]
plt.plot(train)
plt.plot(test)

model=pm.auto_arima(train,start_p=0,d=1,start_q=0,
          max_p=5,max_d=5,max_q=5, start_P=0,
          D=1, start_Q=0, max_P=5,max_D=5,
          max_Q=5, m=12, seasonal=True,
          error_action='warn',trace=True,
          supress_warnings=True,stepwise=True,
          random_state=20,n_fits=50)


prediction = pd.DataFrame(model.predict(n_periods = 30),index=test.index)
prediction.columns = ['predicted_passengers']

plt.figure(figsize=(8,5))
plt.plot(train,label="Training")
plt.plot(test,label="Test")
plt.plot(prediction,label="Predicted")
plt.legend(loc = 'upper left')
#plt.savefig('SecondPrection.jpg')
plt.show()


#plt.show()



#df = [10, 12, 13, 15, 10, 12, 10, 11, 11, 16, 27, 13, 2, 14, 17, 16, 17, 18, 11, 13]
#y = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#y = pd.DataFrame(y)

#df = pd.DataFrame(df)


'''

toPlotList = pd.DataFrame(df)

train, test = train_test_split(df.values, train_size=tsize)

print ("2on3")

model = pm.auto_arima(train, seasonal=False, m=2)

print ("thisnons")
preds = model.predict(test.shape[0])

print ('ok here')
x = np.arange(df.shape[0])
plt.plot(df.values[:tsize], train)
plt.plot(df.values[tsize:], preds)
plt.show()


'''



DATA_PATH = "paris/testLong.csv"

with open(DATA_PATH) as f:
	bitHist = pd.read_csv(DATA_PATH)


dfile = bitHist[['Hash']]
toPlotList = pd.DataFrame(dfile)

totalNum = 1000
#list of hash column
toPlot = toPlotList['Hash'].tolist()

#length of hash - what will be predicted
endPlot = len(toPlot) - totalNum

#starting from 3000 and going until end of training data (green)
toPlotX = toPlot[3000:endPlot]

#from endplot to end (blue)
toPlotY = toPlot[endPlot:]

ysii = toPlotY
xsii = [x for x in range(len(ysii))]

plt.plot(xsii, ysii, color="blue")

#print (len(toPlotX))

varU = len(toPlotX)

ysi = toPlotX
xsi = [x for x in range(len(ysi))]

#plt.plot(xsi, ysi, color="green")
#plt.show()





print ("end")
