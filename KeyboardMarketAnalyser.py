# 
# Original Author: Jack Hall
# Made Public: 26th June 2020
# 

import praw, argparse, re, sys, datetime, os, logging, json, csv
from parse import *
import pandas as pd
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mpl_dates
from currency_converter import CurrencyConverter
import seaborn as sns
logging.basicConfig(level=logging.INFO)

def main():
	# Preparatory
	args = get_args()
	if args.Debug: logging.getLogger().setLevel(logging.DEBUG)
	KeebList = GetListOfKeebs(args)

	# Dataset generation
	if not args.InputData:
		subreddit = GetSubreddit()
		DataDF = GetData(args, subreddit, KeebList)
	else: DataDF = GetExistingData(args.InputData)
	DF = PrepDF(DataDF)

	# Analysis and output
	for Keeb in KeebList:
		if args.PlotData: MakePlots(args, DF, Keeb)
		if args.SaveData: SaveData(args, DF, Keeb)
		if args.DatePredict: DatePredict(args, DF, Keeb)
	logging.info("All finished.")

def DatePredict(Args, DF, Keeb):
	logging.info(80*"-")
	logging.info("Time to predict cost in the future ...")

	# Create output dir
	cwd = os.getcwd() + "/"
	OutputDir = cwd + "Plots/"+Keeb+"/"
	if not os.path.exists(OutputDir): os.makedirs(OutputDir)

	# Select only one keeb from the big DF
	DF = DF.loc[DF['Sale Item'] == Keeb.upper()]
	DF = DF.dropna()
	DF['Post Date'] = DF['Post Date'].map(datetime.datetime.toordinal)

	# Begin Plot:
	x_var_name = "Post Date"
	y_var_name = "Asking Price"
	x_var = DF[x_var_name]
	y_var = DF[y_var_name]

	# Axis and Title labels:
	# KeebName = Keeb.upper()
	# figureTitle = KeebName
	figureTitle = Keeb
	figurexlabel = x_var_name
	figureylabel = y_var_name + " (\$)"
	Prop_array = [figureTitle, figurexlabel, figureylabel]

	# Reshaping
	x_var = x_var.to_numpy()
	x_var = x_var.reshape(-1, 1)
	y_var = y_var.to_numpy()
	y_var = y_var.reshape(-1, 1)

	# Linear regression
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import LinearRegression
	from sklearn.metrics import mean_absolute_error,mean_squared_error
	X_train, X_test, y_train, y_test = train_test_split(x_var, y_var, test_size = 0.1)

	# Fit linear regression
	regr = LinearRegression(fit_intercept=True,copy_X=True)
	regr.fit(X_train, y_train)
	
	# Print the info:
	print('intercept:', regr.intercept_)
	print('slope:', regr.coef_) 
	print("Score:",regr.score(X_test, y_test))

	y_pred = regr.predict(X_test)

	# Extend the line
	XMin = x_var.min(axis=0)[0]
	XMax = x_var.max(axis=0)[0]
	x_extra = np.linspace(XMin,XMax,100,dtype=np.int_)
	x_extra = x_extra.reshape(-1, 1)
	y_extra = regr.predict(x_extra)

	# Evaluate fit
	mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
	rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)
	print("MAE:",mae)
	print("RMSE:",rmse)

	# Transform back to datetime
	X_train = X_train.reshape(-1)
	X_train = pd.Series(X_train)
	X_train = X_train.map(datetime.datetime.fromordinal)
	y_train = y_train.reshape(-1)
	x_extra = x_extra.reshape(-1)
	x_extra = pd.Series(x_extra)
	x_extra = x_extra.map(datetime.datetime.fromordinal)
	y_extra = y_extra.reshape(-1)
	X_test = X_test.reshape(-1)
	X_test = pd.Series(X_test)
	X_test = X_test.map(datetime.datetime.fromordinal)
	y_pred = y_pred.reshape(-1)

	# Plot setup
	plt.rcParams["figure.figsize"] = (12,6)
	plt.style.use("ggplot")

	# Plotting
	plt.scatter(X_train, y_train, color ='g')
	plt.scatter(X_test, y_test, color ='b')
	plt.plot(x_extra, y_extra, color ='r',linestyle='dashed')
	plt.plot(X_test, y_pred, color ='r')

	# Plot adjustments
	plt.xlabel(Prop_array[1],fontsize=14)
	plt.ylabel(Prop_array[2],fontsize=14)
	plt.ylim((0,y_train.max() * 1.05))

	plt.legend(["Training sample", "Testing sample", "Extrapolated range", "Evaluated range"], loc ="lower left")
	# plt.show()

	OutputPlotName = Keeb + "_timeseries_fit" + ".png"
	plt.savefig(OutputDir+OutputPlotName)
	logging.info("Saved to:\t" +OutputDir+OutputPlotName)

	# Prediction
	Input=input("What date do you want to test? (in DD/MM/YYYY) ")
	TestDate=datetime.datetime.strptime(Input,"%d/%m/%Y").date()

	print("Inputted test value:",TestDate)
	TestDate = TestDate.toordinal()
	# print("Transformed:",test_value,type(test_value))
	Array = np.array([TestDate])
	Array = Array.reshape(-1,1)
	print("Predicted asking price ($):",regr.predict(Array))

def MakePlots(Args, DataFrame, Keeb):
	logging.info(80*"-")
	logging.info("Time for plots ...")
	# Select only one keeb from the big DF
	DataFrame = DataFrame.loc[DataFrame['Sale Item'] == Keeb.upper()]
	logging.debug(DataFrame.head())
	logging.debug(DataFrame.describe())
	# So you can copy specific URLs:
	logging.info("URLs in the plot ...")
	pd.set_option('display.max_colwidth', None)
	logging.info(DataFrame["URL"])

	# Create output dir
	cwd = os.getcwd() + "/"
	OutputDir = cwd + "Plots/"+Keeb+"/"
	if not os.path.exists(OutputDir): os.makedirs(OutputDir)

	SoldHist(DataFrame, Keeb, OutputDir)
	PostLocationHist(DataFrame, Keeb, OutputDir)
	AskingPriceHist(DataFrame, Keeb, OutputDir)
	CorrelationPlot(DataFrame, Keeb, OutputDir)
	DatePlot(DataFrame, Keeb, OutputDir)
	# ItemPlotMPL(DataFrame, List_of_Keebs, OutputDir)
	return DataFrame

def SoldHist(DF, Keeb, OutputDir):
	# Prep DF for plot:
	LocationDensity = DF['Sold'].value_counts()  
	LocationDensityDF = LocationDensity.reset_index()
	LocationDensityDF.columns = ['Sold', 'frequency']

	logging.info("Producing histogram of sold ... ")
	g = sns.barplot(data=LocationDensityDF, x='Sold', y='frequency', palette="viridis")
	plt.show()

	OutputPlotName = Keeb + "_sold_dist" + ".png"
	g.figure.savefig(OutputDir+OutputPlotName)
	logging.info("Saved to:\t" +OutputDir+OutputPlotName)

def PostLocationHist(DF, Keeb, OutputDir):
	# Prep DF for plot:
	LocationDensity = DF['Post Location'].value_counts()  
	LocationDensityDF = LocationDensity.reset_index()
	LocationDensityDF.columns = ['Post Location', 'frequency']

	logging.info("Producing histogram of post locations ... ")
	g = sns.barplot(data=LocationDensityDF, x='Post Location', y='frequency', palette="viridis")

	# Axis again
	g.set_xticklabels(g.get_xticklabels(), rotation=90, size=5)

	plt.show()

	OutputPlotName = Keeb + "_post_location_dist" + ".png"
	g.figure.savefig(OutputDir+OutputPlotName)
	logging.info("Saved to:\t" +OutputDir+OutputPlotName)

def AskingPriceHist(DF, Keeb, OutputDir):
	logging.info("Producing histogram of asking price ... ")
	
	# Plotting
	ax = sns.histplot(data=DF, x="Asking Price", hue="Sold", kde=True)
	sns.kdeplot(data=DF, x="Asking Price", color='crimson', ax=ax)

	# x axis scale
	ax.set_xlim([0, DF["Asking Price"].max() * 1.05])
	plt.xlabel("Asking price (\$)",fontsize=14)
	plt.show()

	OutputPlotName = Keeb + "_asking_price_dist" + ".png"
	ax.figure.savefig(OutputDir+OutputPlotName)
	logging.info("Saved to:\t" +OutputDir+OutputPlotName)

def DatePlot(DF, Keeb, OutputDir):
	logging.info("Making plot of asking price vs date posted for keeb:\t"+Keeb)

	DF = DF.dropna()

	# Begin Plot:
	x_var_name = "Post Date"
	y_var_name = "Asking Price"
	x_var = DF[x_var_name]
	y_var = DF[y_var_name]

	# Axis and Title labels:
	# KeebName = Keeb.upper()
	# figureTitle = KeebName
	figureTitle = Keeb
	figurexlabel = x_var_name
	figureylabel = y_var_name + " (\$)"
	Prop_array = [figureTitle, figurexlabel, figureylabel]

	# Plot styles
	plt.rcParams["figure.figsize"] = (12,6)
	plt.style.use("ggplot")

	# Code to plot:
	# g = sns.scatterplot(data=DF, x=x_var, y=y_var,hue="Sold")
	# g = sns.relplot(data=DF, x=x_var, y=y_var,hue="Sold")
	g = sns.stripplot(x=x_var,y=y_var,data=DF,hue="Sold", jitter=False)
	
	g.figure.suptitle("Sales of "+Prop_array[0],fontsize=16)
	g.figure.set_size_inches(12,6)
	plt.subplots_adjust(bottom=0.26)

	# Axes labels
	plt.xlabel(Prop_array[1],fontsize=14)
	plt.ylabel(Prop_array[2],fontsize=14)
	# Y axis scale
	ax = plt.gca()
	ax.set_ylim([0, y_var.max() * 1.05])
	# X Axis ticks
	# plt.gcf().autofmt_xdate()
	g.set_xticklabels(g.get_xticklabels(), rotation=90, size=5)
	# # date_format = mpl_dates.DateFormatter('%b, %d %Y')
	# # ax.xaxis.set_major_formatter(date_format)

	# Final matter:
	plt.show()
	logging.info(plt.figure())
	logging.info(plt.axes())

	OutputPlotName = Keeb + "_sales_timeseries" + ".png"
	g.figure.savefig(OutputDir+OutputPlotName)
	logging.info("Saved to:\t" +OutputDir+OutputPlotName)

def CorrelationPlot(DF, Keeb, OutputDir):
	logging.info("Producing correlation matrix plot ... ")
	g = sns.heatmap(DF.corr(numeric_only = True))
	plt.show()

	OutputPlotName = Keeb + "_correlation_matrix" + ".png"
	g.figure.savefig(OutputDir+OutputPlotName)
	logging.info("Saved to:\t" +OutputDir+OutputPlotName)

def SaveData(Args, DataFrame, Keeb):
	cwd = os.getcwd() + "/"
	OutputDir = cwd + "csv_data/"
	if not os.path.exists(OutputDir): os.makedirs(OutputDir)

	# NewKeebName = ("_").join(Keeb.split(" "))
	NewKeebName = Keeb
	OutputDataName = NewKeebName + ".csv"
	DataFrame.to_csv(OutputDir + OutputDataName, index=False)
	logging.info("Outputed file to:\t" + OutputDir + OutputDataName)

def GetData(Args, subreddit, List_of_Keebs):
	List_Of_Output_Dicts = []
	for Keeb in List_of_Keebs:
		search_subreddit = subreddit.search("[H] " + Keeb + " [W] PayPal", limit=Args.DataLimit)
		Keeb_Renamed = Keeb.upper()
		for submission in search_subreddit:
			# Loop over posts in the subreddit
			if CheckGoodSubmission(submission, Keeb_Renamed):
				if not submission.stickied:
					# Stickied is weekly stuff
					PrintSubmissionInfo(submission)

					# Need to parse title to find whats been sold
					Location, HaveObjects, WantObjects = ParseTitle(submission)
					# Need to parse post body to find prices
					ObjectDict = ParsePostBody(submission,HaveObjects,WantObjects,Location, Keeb_Renamed)
					
					PostDate = PrintOutputDict(ObjectDict, submission, Location)
					if Args.DebugKeeb: input()

					# Now we have most of the data for each sale, append the last parts
					for key, value in ObjectDict.items():
						List_Of_Output_Dicts.append(
							{
								'Post Location': Location,
								'Post Date': PostDate,
								'User': submission.fullname,
								'URL': submission.url,
								'Sale Item':  key,
								'Asking Price': value[0],
								'Sold': value[1]
							}
						)
	return pd.DataFrame(List_Of_Output_Dicts)

def PrepDF(DataFrame):
	# Create a dataframe from the output dictionary
	DataFrame = DataFrame.sort_values(by='Post Date',ascending=True)
	DataFrame['Post Date'] = pd.to_datetime(DataFrame['Post Date'], format="%Y/%m/%d")
	DataFrame = DataFrame.convert_dtypes()
	DataFrame['Sold'] = DataFrame['Sold'].astype('bool')
	DataFrame['Asking Price'] = DataFrame['Asking Price'].astype('float64')
	# Get names of indexes for which column Age has value 30
	indexNames = DataFrame[ DataFrame['Asking Price'] == 1 ].index
	# Delete these row indexes from dataFrame
	DataFrame.drop(indexNames , inplace=True)
	# DataFrame = DataFrame.dropna()

	# Output:
	logging.info(80*"-")
	logging.info("\t\t\tDATAFRAME SUMMARY ")
	logging.info("Technical info of the dataframe ...")
	logging.info(DataFrame.info())
	logging.info("First 5 entries ...")
	logging.info(DataFrame.head())
	logging.info("The number of posts found is:\t"+str(len(DataFrame.index)))
	logging.info("Correlation matrix for the data ...")
	logging.info(DataFrame.corr(numeric_only = True))
	logging.info("Stats for asking prices ...")
	logging.info(DataFrame.describe())
	logging.info("How much data is in each column ...")
	logging.info(DataFrame.count(0))
	logging.info("How many are sold ...")
	logging.info(DataFrame['Sold'].value_counts(sort = True, ascending  = True))
	logging.info("How many are sold as a ratio ...")
	logging.info(DataFrame['Sold'].value_counts(sort = True, ascending  = True, normalize=True))
	logging.info("Location density of the posts ...")
	LocationDensity = DataFrame['Post Location'].value_counts(sort = True, ascending  = False)
	logging.info(LocationDensity)
	return DataFrame

def GetListOfKeebs(Args):
	# If you want to hard-code a list of many, do it here:
	Custom_List_Of_Keebs = []
	if Custom_List_Of_Keebs: List_of_Keebs = Custom_List_Of_Keebs
	else: List_of_Keebs = Args.Keeb
	return List_of_Keebs

def GetSubreddit():
	F = open("login.json")
	JSONF = json.load(F)
	reddit = praw.Reddit(user_agent=JSONF["user_agent"], client_id=JSONF["client_id"],
	                     client_secret=JSONF["client_secret"], username=JSONF["username"],
	                     password=JSONF["password"])
	subreddit = reddit.subreddit("mechmarket") 
	F.close()
	return subreddit

def PrintSubmissionInfo(Submission):
	logging.info(80*"-")
	logging.info("\t\t\t\t SUBMISSION:")
	logging.info("Title:\t" + str(Submission.title))
	logging.debug("Ups:\t" + str(Submission.ups) + "\t\tDowns:\t" + str(Submission.downs))
	logging.info("URL:\t" + str(Submission.url))
	logging.info(80*"-")

def PrintOutputDict(Dictionary, Submission, Location):
	logging.info(80*"-")
	logging.info("OUTPUT DATA:")
	SubDate = GetDate(Submission)
	logging.info("Post Location:\t\t" + str(Location))
	logging.info("Post Date:\t\t" + str(SubDate))
	logging.info("User:\t\t\t" + str(Submission.fullname))
	logging.info("KEEBS:")
	for key, value in Dictionary.items():
		logging.info("Keyboard:\t\t" + str(key))
		logging.info("Asking Price:\t\t" + str(value[0]))
		logging.info("Sold:\t\t\t" + str(value[1]))
	logging.info(80*"-")
	return SubDate

# def ItemPlotMPL(DF, List_of_Keebs, OutputDir):
# 	# Actual Plot:
# 	x_var_name = "Post Date"
# 	y_var_name = "Asking Price"
# 	x_var = DF[x_var_name]
# 	y_var = DF[y_var_name]

# 	# Axis and Title labels:
# 	KeebName = List_of_Keebs[0].upper()
# 	figureTitle = KeebName
# 	figurexlabel = x_var_name
# 	figureylabel = y_var_name + " (\$)"
# 	Prop_array = [figureTitle, figurexlabel, figureylabel]

# 	# Plot settings
# 	plt.rcParams["figure.figsize"] = (12,6)
# 	plt.style.use("ggplot")

# 	# Code to plot:
# 	colours = y_var * (100. / y_var.max())
# 	plt.scatter(x_var,y_var,c=colours,cmap="RdYlBu",alpha=0.9)

# 	plt.subplots_adjust(bottom=0.26)

# 	# Axes
# 	plt.xlabel(Prop_array[1],fontsize=14)
# 	plt.ylabel(Prop_array[2],fontsize=14)
# 	ax = plt.gca()
# 	ax.set_ylim([0, y_var.max() * 1.05])
# 	plt.gcf().autofmt_xdate()
# 	date_format = mpl_dates.DateFormatter('%b, %d %Y')
# 	ax.xaxis.set_major_formatter(date_format)

# 	# Other
# 	plt.title("Sales of "+Prop_array[0])

# 	# Final matter:
# 	plt.show()
# 	logging.info(plt.figure())
# 	logging.info(plt.axes())

# 	# Output
# 	NewKeebName = ("_").join(KeebName.split(" "))
# 	OutputPlotName = NewKeebName + ".png"
# 	# g.figure.savefig(OutputDir+OutputPlotName)
# 	logging.info("Saved to:\t" +OutputDir+OutputPlotName)


def CheckGoodSubmission(Submission, Keyboard):
	Title = Submission.title.upper()
	# Get lots of index locations
	HaveLocationIndex = Title.find("[H]")
	WantLocationIndex = Title.find("[W]")
	FirstPayPalLocationIndex = Title.find("PAYPAL")
	SecPayPalLocationIndex = Title.rfind("PAYPAL")
	if "TGR" in Keyboard: tmp_keeb_name = Keyboard.split("TGR ")[1]
	else: tmp_keeb_name = Keyboard
	KeebLocationIndex = Title.find(tmp_keeb_name.upper())
	TradesLocationIndex = Title.find("TRADE")
	if KeebLocationIndex > WantLocationIndex:
		# [W] Keeb
		return False
	if KeebLocationIndex > TradesLocationIndex and TradesLocationIndex != -1:
		# Trades, Keeb
		return False
	if TradesLocationIndex > WantLocationIndex:
		# [W] Trades
		return False
	else:
		# These are the good ones
		return True

def GetDate(submission):
	time = submission.created
	WholeDTObj = datetime.datetime.fromtimestamp(time)
	date = WholeDTObj.date()
	# Other options:
	# logging.info(blah.timestamp())
	# logging.info(blah.year)
	# logging.info(blah.month)
	# logging.info(blah.day)
	return date

def ParseTitle(Submission):
	# Need to update this to a regular expression
	Title = Submission.title.upper()

	pattern = re.compile(r'\[(.*)\].*(\[H\])(.*)(\[W\])(.*)')
	match = pattern.search(Title)

	Location = match.group(1)
	HaveObjects = match.group(3).strip()
	WantObjects = match.group(5).strip()

	if "(" in HaveObjects or "+" in HaveObjects:
		# Found a title with brackets in (THESE ARE PROBLEMATIC)
		HaveObjects = HaveObjects.replace("(","\(")
		HaveObjects = HaveObjects.replace(")","\)")
		HaveObjects = HaveObjects.replace("+","")
	return Location, HaveObjects, WantObjects

def CheckIfSold(Submission, PostBody):
	if Submission.link_flair_text:
		# If it has a post flair, it might already say its sold
		if "Sold" in Submission.link_flair_text:
			logging.debug("Submission was sold, assuming it was asking price")
			return True
	# Search for Sold in body
	logging.debug("Flair has not been updated, checking in the body")
	if search("**SOLD**",PostBody) is not None: return True
	elif search("SOLD ",PostBody) is not None: return True
	elif search("SOLD",PostBody) is not None:
		if search("SOLDER",PostBody) is not None: return False
		else: return True
	else: return False

def CurrencyConversion(Submission, Match, Currency):
	c = CurrencyConverter(fallback_on_missing_rate=True)

	Price = Match.group(1)
	AskingPrice = re.sub(Currency, '', Price)
	AskingPrice = float(AskingPrice)
	# if Currency == "\€": AskingPrice = c.convert(AskingPrice, 'EUR', 'USD',date=GetDate(Submission))
	# elif Currency == "\£": AskingPrice = c.convert(AskingPrice, 'GBP', 'USD',date=GetDate(Submission))
	if Currency == "\€": AskingPrice = c.convert(AskingPrice, 'EUR', 'USD')
	elif Currency == "\£": AskingPrice = c.convert(AskingPrice, 'GBP', 'USD')
	return AskingPrice

def PriceCheck(Text, Currency, Keyboard = "None", SoldCheck = False):
	if Keyboard == "None": SearchStr = '('+Currency+'(\d+)|(\d+)'+Currency+')'
	# elif Keyboard != "None" and SoldCheck == False: SearchStr = Keyboard+'.*'+Currency+'(\d{1,9})'
	elif Keyboard != "None" and SoldCheck == False: SearchStr = Keyboard+'.*'+'('+Currency+'(\d+)|(\d+)'+Currency+')'
	else: SearchStr = Keyboard+'.*'+Currency+'(\d{1,9}).*SOLD'
	pattern = re.compile(r''+SearchStr)
	match = pattern.search(Text)
	return match

def SearchForPrice(Submission, PostBody, Item = "None"):
	Currencies = ["\$", "\€", "\£"]
	for Currency in Currencies:
		if PriceCheck(PostBody, Currency, Item) is not None: return PriceCheck(PostBody, Currency, Item), Currency
	return None, None

def ParsePostBody(Submission, HaveObjects, WantObjects, Location, Keyboard):
	logging.debug("PARSING POST BODY LOG:\n")
	PostBody = Submission.selftext.upper()
	
	ObjectDict = {}
	if len(HaveObjects.split(",")) == 1:
		logging.debug("Only has one item for sale so finding the price ...")
		# Only one object so need to find the price
		ListInfo = []
		logging.debug("Searching for: " + HaveObjects)
		logging.debug("In text:\n"+PostBody)
		Match,Currency = SearchForPrice(Submission, PostBody)
		if Match is not None:
			Price = CurrencyConversion(Submission, Match,Currency)
			SellOutcome = CheckIfSold(Submission, PostBody)
			logging.debug("Info found ... Price: "+str(Price)+" Currency: "+Currency+" Sell outcome: "+str(SellOutcome))
		else:
			Price = None
			SellOutcome = False
			logging.debug("Not found price so putting in nans ... ")
		
		ListInfo.append(Price)
		ListInfo.append(SellOutcome)
		ObjectDict.update({Keyboard:ListInfo})
		return ObjectDict
	else:
		# For posts with multiple items for sale
		logging.debug("Has multiple items for sale so looping over them ...")
		for Object in HaveObjects.split(","):
			Object = Object.strip()
			ListInfo = []
			logging.info("Searching for:\t\t" + Object)
			Match,Currency = SearchForPrice(Submission, PostBody, Object)
			if Match is not None:
				Price = CurrencyConversion(Submission, Match,Currency)
				SellOutcome = CheckIfSold(Submission, PostBody)
				logging.debug("Info found ... Price: "+str(Price)+" Currency: "+Currency+" Sell outcome: "+str(SellOutcome))
			else:
				Price = None
				SellOutcome = False
				logging.debug("Not found price so putting in nans ... ")
			ListInfo.append(Price)
			ListInfo.append(SellOutcome)
			ObjectDict.update({Object:ListInfo})
		return ObjectDict

def GetExistingData(InputData):
	return pd.read_csv(InputData)

def get_args():
	args = argparse.ArgumentParser()
	args.add_argument('--Keeb', type=str, nargs='+', default=["polaris"], help='Name of keebs you would like to search for - seperate by space e.g: --Keeb polaris aella.')
	args.add_argument('--SaveData', action='store_true', help='When youre happy to export your dataframe, turn this on')
	args.add_argument('--InputData', type=str, default=None, help='Use an existing dataset youve generated.')
	args.add_argument('--PlotData', action='store_true', help='Get a sales version of the final DF, for single keebs only.')
	args.add_argument('--DatePredict', action='store_true', help='Use timeseries plot to predict price in the future')
	args.add_argument('--DebugKeeb', action='store_true', help='When gathering data, pause after each keeb to check the log.')
	args.add_argument('--DataLimit', type=int, default=40000, help='Restrict the number of listings it will perform per keyboard.')
	args.add_argument('--Debug', action='store_true', help="Changes the log level to debug")
	return args.parse_args()

if __name__ == '__main__': main()
