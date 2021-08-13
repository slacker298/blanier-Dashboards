# Short Interest Dashboard 

## https://blaniershortinterest.herokuapp.com/


This dashboard accumulates short interest data and graphs it out in an easy to see manner.  

Short Interest data is hard to come by - unsurprisingly, many would rather such information not be easily obtainable.  Official open short interest is only required to be reported every two weeks.  And accsessability to such data varies depending on the exchange the security is traded on. 

What is available is the official short interest volume reported everyday by FINRA.  This code uses FINRA's api (https://api.finra.org) to get that data and graph it out. 

Securities traded on NASDAQ have easily obtainable "official" short interest.  Every pull we check if the symbol has data on (http://www.nasdaqtrader.com/Trader.aspx?id=ShortInterest).  If so we pull the data from a simple http request and plot the data accordingly. 

Securuites traded on the NYSE do not have such simple methods of getting official short interest data. 

