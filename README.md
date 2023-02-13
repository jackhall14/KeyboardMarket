# Keyboard Market Analyser for subreddit: r/mechmarket

## Purpose
A piece of code that allows one to analyse the secondary keyboard market via the subreddit: r/mechmarket.

You might want information on the prices of a keycap set you're looking to buy and/or sell and want to know if it's expensive/cheap, will it sell at a given price?

You might want to gather a load of information on some high-end keyboards and perform you're own data analysis for investments.

Whatever item you can search into r/mechmarket, you can search for it here quickly and have a record of it for you to play with.
	
## Setup
To run this project:

```
$ cd <Some directory you'd like to work from>
$ git clone https://github.com/jackhall14/KeyboardMarket.git
$ You'll need loads of dependencies so get download each one via pip or use a virtual environment like anaconda and get downloading using pip. Most are included in the bash file `source requirements.sh`
$ For the reddit scraper to work, it'll need some reddit account details so put them into `login.json`. Best reference for how to get this is this youtube video: https://www.youtube.com/watch?v=NRgfgtzIhBQ
$ Now you can do:
$ python KeyboardMarketAnalyser.py
```

## Usage:
1. `--PlotData` Argument, this creates a dataframe of sales of (and only of) the item from `--Keeb` as well as plotting them as a time series. This option is good for those that are only interesting in prices of a specific item in the keeb community or those who want a quick time series plot. As an example you might run:
```bash
python KeyboardMarketAnalyser.py --Keeb Polaris --PlotData
```
And you'll get several plots which will be automatically saved in `/Yourpath/Plots/POLARIS/*.png`. The first plot you will get is a visualisation of the correlation matrix:

![Image of correlation matrix](https://github.com/jackhall14/KeyboardMarket/blob/master/Plots/polaris/Polaris_correlation_matrix.png)

Which can also be found presented in the terminal:
```bash
INFO:root:Correlation matrix for the data ...
INFO:root:              Asking Price      Sold
Asking Price      1.000000  0.098962
Sold              0.098962  1.000000
```

The next three plots are essentially histograms of the main three variables:

![Image of sold hist](https://github.com/jackhall14/KeyboardMarket/blob/master/Plots/polaris/Polaris_sold_dist.png)
![Image of asking price hist](https://github.com/jackhall14/KeyboardMarket/blob/master/Plots/polaris/Polaris_asking_price_dist.png)
![Image of post location hist](https://github.com/jackhall14/KeyboardMarket/blob/master/Plots/polaris/Polaris_post_location_dist.png)

The final plot created is a time series distribution of the asking prices:

![Image of Polaris](https://github.com/jackhall14/KeyboardMarket/blob/master/Plots/polaris/polaris_sales_timeseries.png)

All the information plotted and more can be found directly printed in the terminal:
```bash
INFO:root:First 5 entries ...
INFO:root:   Post Location  Post Date       User  ... Sale Item Asking Price  Sold
8          US-TX 2020-04-15  t3_g210i0  ...   POLARIS        340.0  True
...

[5 rows x 7 columns]
INFO:root:The number of posts found is:	377
INFO:root:Correlation matrix for the data ...
INFO:root:              Asking Price      Sold
Asking Price      1.000000  0.098962
Sold              0.098962  1.000000
INFO:root:Stats for asking prices ...
INFO:root:       Asking Price
count    375.000000
mean     285.120000
std      229.103937
min        8.000000
25%       80.000000
50%      225.000000
75%      477.500000
max     1250.000000
INFO:root:How much data is in each column ...
INFO:root:Post Location    377
Post Date        377
...
Sold             377
dtype: int64
INFO:root:How many are sold ...
INFO:root:True     139
False    238
Name: Sold, dtype: int64
INFO:root:How many are sold as a ratio ...
INFO:root:True     0.3687
False    0.6313
Name: Sold, dtype: float64
INFO:root:Location density of the posts ...
INFO:root:US-CA    80
US-NY    54
...
```


2. `--SaveData` Argument, this outputs (via a csv file) a dataframe of sale information from mechmarket posts. This is good if you want to perform your own statistical analyses or machine learning. For example I might run:
```bash
python KeyboardMarketAnalyser.py --Keeb TGR Jane v2 --SaveData
```
Note, because some posts sell more than one item, we take all sale items regardless as we can filter later if need be so the output dataframe can be like:
```bash
  Post Location   Post Date       User                                                URL                 Sale Item  Asking Price   Sold
0          CA-ON  2019-09-07  t3_d0lnbr  https://www.reddit.com/r/mechmarket/comments/d...               TGR JANE V2        2400.0   True
1          US-CA  2020-06-17  t3_hadwwn  https://www.reddit.com/r/mechmarket/comments/h...              GMK NAUTILUS         350.0  False
...
Outputed file to:
/Yourpath/csv_data/TGR_Jane_v2.csv
Finished.
```

### Additional Argument Options
1. `--DebugKeeb` Which as the code searchs through and finds a good match, you can check it, use it for diagnostics plus other benefits you can think of
2. `--DataLimit` Which limits the number of searchs, so set it to 10 for example if you're debugging a problem as the reddit praw has a search data limit request
2. `--Debug` Changes the logging output from info to debug for more information

## Technical Info of the Code

So this a wrapper of the reddit praw api. It searches for [H]<Some item> [W] Paypal, and filters only the responses that are looking for payments over trades. I then take parse information from the title and the post body using regular expressions. For a post with multiple sales, we store all the sale items as we can remove them later if we don't need them. Note, for the plots, I write the price label in USD, which it most is but I have included a currency converter for any price adjustments from EUR and GBP too.
