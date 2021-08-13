# Traffic & Weather 

Another personal project - inspired by being stuck in Austin traffic on i35 on a nice day.  I wondered if the weather (sunny, rainy, etc.) had any effect on how much traffic there would be.  
The data was gathered from open source datasets on .gov websites.

Inside the notebooks there is much fun had modeling and exploring data, but the results can most clearly be seen in one of the linear regressions run in Traffic_weather3_noscaling.ipynb

Using only dummy variables for the day of week and month of the year, we can achieve an R-squared of 0.75
Meaning, 75% of the variation in traffic can be explained by day of the week and month of the year.  This result, while not exciting, is very easy to believe.  Weather be damned, people are going to work on the weekdays, and going to yearly events throughout the year.