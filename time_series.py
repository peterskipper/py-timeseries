import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime
import matplotlib.pyplot as plt

df = pd.read_csv('LoanStats3b.csv', header=1, low_memory=False)
df['issue_d_format'] = pd.to_datetime(df['issue_d'])
ts = df.set_index(df['issue_d_format'])

# Group by year and month, uniquely
year_month_summary = ts.groupby(lambda x: x.year*100+x.month).count()

loan_count_summary = year_month_summary['issue_d_format']
plt.plot(loan_count_summary)
plt.title('Loan Count by Month, 2012-13')
plt.savefig('loan_count_series.png')

# Series is clearly not stationary
print 'The series is not stationary and will be differenced...'
differenced_loan = loan_count_summary.diff(periods=1)
differenced_loan.dropna(inplace=True)
sm.graphics.tsa.plot_acf(differenced_loan, alpha=0.05)
plt.title('ACF Differenced Series')
plt.savefig('differenced_acf.png')
sm.graphics.tsa.plot_pacf(differenced_loan, alpha=0.05)
plt.title('PACF Differenced Series')
plt.savefig('differenced_pacf.png')

print 'There are no significant lags after differencing'
print ('Open "loan_count_series.png", "differenced_acf.png" & '
       '"differenced_pacf.png" to see results')
