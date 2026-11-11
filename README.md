# ChannelGuard: predicting digital banking outages before they happen

Building AI course project

## Summary

ChannelGuard looks at the monitoring data banks already collect (response times, errors, server load) and estimates the probability that a digital channel like mobile banking or ATMs goes down in the next hour, so IT can fix it before customers notice.

## Background

I work as a deployment engineer setting up IT monitoring tools (ManageEngine OpManager and Applications Manager) for banks and government clients in Oman. One thing I keep seeing is that monitoring is mostly reactive: an alert fires when something is already broken, the team rushes to fix it, and afterwards someone writes the downtime report for the central bank.

The problems:
* outages are noticed only after customers are already affected
* simple threshold alerts are noisy, so people start ignoring them
* every minute of mobile or internet banking downtime costs money and trust, and has to be reported to the regulator
* the warning signs (slower responses, rising error rate, memory creeping up) are often in the data long before the crash

I already build the downtime reports. My motivation is to move from explaining outages to preventing them.

## How is it used?

* It runs next to the existing monitoring. Every 5 minutes it takes the latest metrics for each channel and gives a risk score from 0 to 100%.
* If the risk goes above a set level (for example 70%), it opens a ticket or alerts the on-call engineer, together with the metrics that pushed the risk up.
* The NOC team gets a simple view of all channels sorted by risk.

Users are NOC operators and IT operations managers. Bank customers benefit indirectly, and the compliance team has fewer incidents to report. The prediction has to be explainable: an engineer won't act on a black box at 3 am.

## Data sources and AI methods

| Data | Source |
| ----------- | ----------- |
| Response time, availability and error rate per channel | Applications Manager (APM) |
| CPU, memory, disk and network of servers and databases | OpManager |
| Past incidents with start and end times | Service desk tickets and downtime reports |
| Calendar features: salary days, holidays, Ramadan | Built manually |

Each 5-minute window is labelled 1 if an outage started within the next 60 minutes and 0 otherwise, which makes this a supervised classification problem.

* **Logistic regression** first: it outputs a probability directly and its weights are easy to explain.
* **k-nearest neighbors** to compare: "has the system looked like this before an outage?"
* A small **neural network** later, if there is enough data.
* **No random splits.** Train on older months and test on newer ones, and use cross-validation to avoid overfitting.
* Outages are rare (well under 1% of windows), so plain accuracy is misleading. A model that always says "no outage" would score 99%. I'd measure how many outages were caught and how many false alarms happen per week.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# X: one row per 5-minute window -> [avg_response_ms, error_rate, cpu, memory, tps, is_salary_day]
# y: 1 if an outage started within the next 60 minutes, otherwise 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)

risk = model.predict_proba(X_test)[:, 1]   # chance of an outage in the next hour
alerts = risk > 0.7
```

## Challenges

* Few positive examples. Outages are rare, so it needs a long history or data from several banks.
* Banking data is sensitive, so the whole thing should run on-premise inside the bank.
* Some outages come from outside, such as a telecom link, a third-party payment switch or power, and won't show up in the metrics.
* Too many false alarms would bring back the alert fatigue it's trying to fix.
* Systems change after upgrades, so the model needs regular retraining.
* It predicts but doesn't fix anything. People still make the decisions.

## What next?

* Pilot on one channel (mobile banking) with historical data and compare it with the current threshold alerts.
* Show which metric is driving the risk, as a first hint at the root cause.
* Package it as an add-on for monitoring tools so other clients can use it.
* Later, suggest or trigger safe actions (restart a service, switch to backup) with human approval.

To get there I'd need anonymised historical data, a bank willing to pilot it, and help from someone with more data science experience.

## Acknowledgments

* [Building AI](https://buildingai.elementsofai.com/) course by Reaktor, University of Helsinki and MinnaLearn
* [scikit-learn](https://scikit-learn.org/) (BSD license), used in the example code
* Inspired by my day-to-day work with ManageEngine monitoring tools
