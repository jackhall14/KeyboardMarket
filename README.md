# KeyboardMarket

## Purpose
A piece of code that allows one to analyse the secondary keyboard market via the subreddit: r/mechmarket.

You might want information on the prices of a keycap set you're looking to buy and/or sell and want to know if it's expensive or cheap.

You might want to gather a load of information on some high-end keyboards and perform you're own data analysis for investments.

Whatever item you can search into r/mechmarket, you can search for it here quickly and have a record of it for you to play with.
	
## Setup
To run this project:

```
$ cd <Some directory you'd like to work from>
$ git clone https://github.com/jackhall14/KeyboardMarket.git
$ You'll need loads of dependencies so get downloading via pip or use a virtual environment like anaconda and get downloading using pip
$ I have already put some of them in requirements.sh
$ L250ish, You need to put in some details for the reddit praw api. Best reference for this, is this youtube video: https://www.youtube.com/watch?v=NRgfgtzIhBQ
$ python KeyboardMarketAnalyser.py
```

## Usage:
1. `--PlotData` Argument, this creates a dataframe of sales of (and only of) the item from `--Keeb` as well as plotting them as a time series. This option is good for those that are only interesting in prices of a specific item in the keeb community or those who want a quick time series plot. As an example you might run:
```bash
python KeyboardMarketAnalyser.py --Keeb Polaris --PlotData
```
And you'll get a plot which will be automatically saved in `/Yourpath/Plots/POLARIS/POLARIS.png`.
![Image of Polaris](https://github.com/jackhall14/KeyboardMarket/blob/master/POLARIS.png)

As well as a dataframe:
```bash
   Post Location   Post Date       User                                                URL Sale Item  Asking Price   Sold
72         CA-ON  2020-03-18  t3_fk7pkv  https://www.reddit.com/r/mechmarket/comments/f...   POLARIS         450.0   True
64         US-MD  2020-04-02  t3_fth9hz  https://www.reddit.com/r/mechmarket/comments/f...   POLARIS         325.0   True
...
<Price info:>
       Asking Price
count     24.000000
mean     420.000000
std      275.984404
min       45.000000
<Direct urls to go a look at the item:>
72     https://www.reddit.com/r/mechmarket/comments/fk7pkv/caon_h_polaris_60_kit_and_tofu_60_case_w_paypal/
64                        https://www.reddit.com/r/mechmarket/comments/fth9hz/usmd_h_ai03_polaris_w_paypal/
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

So this a wrapper of the reddit praw api. It searches for [H]<Some item> [W] Paypal, and filters only the responses that are looking for payments over trades. I then take parse information from the title and the post body using regular expressions. For a post with multiple sales, we store all the sale items as we can remove them later if we don't need them. Note, for the plots, I write the price label in USD, which it most likely is but this is inaccurate and could be in euros or any other currency for that matter.
